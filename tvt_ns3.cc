// =============================================================
#include <csignal>
#include <cstring>
#include <cstdio>
#include "v2x-kpi.h"
#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/config-store-module.h"
#include "ns3/config-store.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/log.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/stats-module.h"
#include "ns3/nr-sl-helper.h"
#include "ns3/lte-sl-tft.h"
#include "ns3/eps-bearer.h"
#include <iomanip>
#include "ns3/nr-phy-rx-trace.h"
#include "ns3/three-gpp-v2v-propagation-loss-model.h"  // [RADIO-DIAG] NLOSv 추가손실 트레이스용
#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <map>
#include <unordered_map>
#include <execinfo.h>
#include <unistd.h>
#include <ctime>
#include <fstream>
// ── ns3-ai 공유 메모리 인터페이스 ─────────────────────────────────────────
#include "tvt-ai.h"
#include "ns3/ai-module.h"
#include <sstream>   // E-9: --loadTransitionsSec parsing
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("NrV2xGnbRsu");

// ============================================================
// [TxPower] 링크별 송신 전력 상수 (dBm)
// NrGnbPhy::TxPower / NrUePhy::TxPower 속성은 둘 다 "Transmission power
// in dBm" (DoubleValue, Watt 변환 없음) — dBm 값을 그대로 DoubleValue()에
// 전달하면 된다.
// ============================================================
static const double UU_UE_TXPOWER_DBM   = 23.0;  // Uu 업링크 (carNode UE → gNB)
static const double UU_GNB_TXPOWER_DBM  = 46.0;  // Uu 다운링크 (gNB → UE)
static const double PC5_UE_TXPOWER_DBM  = 23.0;  // PC5 업링크 (carNode/activeVeh UE)
// SIRIUS 하드웨어 물리적 최대 23dBm이나, 현재 펌웨어에서 실제 설정 가능한
// 값은 0~20dBm(2dBm 단위)뿐 — 23dBm은 TBD(미지원)이므로 20dBm으로 설정.
static const double PC5_RSU_TXPOWER_DBM = 20.0;  // PC5 다운링크 (RSU)

// ============================================================
// [A] Duplication degree state
// ============================================================
// 현재 적용 중인 packet duplication degree. TvtAiAction 과 같은 값 공간이다.
//   0 = 0%   duplication
//   1 = 25%  duplication
//   2 = 50%  duplication
//   3 = 75%  duplication
//   4 = 100% duplication
// 이것은 경로 선택이 아니다. 모든 logical packet 은 primary(Uu)로 나가고,
// 선택된 비율만큼의 logical packet 에만 secondary(PC5) 복제본이 추가된다.
// traffic splitting/striping/load balancing 이 아니다. Uu 와 PC5 연결은
// 어느 degree 에서도 항상 함께 유지된다.
static int g_currentAction = TVT_AI_ACTION_DUP_0;

static const char*
TvtActionName(int action)
{
    switch (action)
    {
    case TVT_AI_ACTION_DUP_0:   return "dup0";
    case TVT_AI_ACTION_DUP_25:  return "dup25";
    case TVT_AI_ACTION_DUP_50:  return "dup50";
    case TVT_AI_ACTION_DUP_75:  return "dup75";
    case TVT_AI_ACTION_DUP_100: return "dup100";
    default:                    return "?";
    }
}

// -- ToD 트래픽 PDR (원격제어) --
static uint32_t g_txTodPackets = 0;
static uint32_t g_rxTodPackets = 0;

// -- CP 트래픽 PDR (협력인지) --
static uint32_t g_rxCpPackets = 0;

// -- 핸드오버 파라미터 (main에서 설정) --
static double g_simEndTime        = 30.0;   // 시뮬레이션 종료 시각 (s)

// ============================================================
// [C] 전역 변수
// ============================================================
uint64_t g_totalRxBitsSl = 0;
uint32_t g_rxPacketsSl   = 0;

// ---- [C1] ToD 전송 — plain UDP DL (server→carNode) ----
static Ipv4Address             g_ue1Addr;           // carNode Uu IP (DL Uu 직접 목적지)
static Ipv4Address             g_rsuUuAddr;         // RSU Uu IP (DL PC5 모드 목적지)
static Ipv4Address             g_ue1V2v0Addr;       // carNode v2v0 IP (RSU 릴레이 수신처)
static Ipv4Address             g_rsuV2v0Addr;       // RSU v2v0 IP (DL PC5 모드 목적지)

static uint32_t                g_todSeqNum  = 0;
static uint32_t                g_todPktSize        = 1000; //원래 4,000에서 1,000으로
// ns3-ai 공유 메모리 세그먼트 이름. 기본값은 기존 동작과 bit 단위로 동일하다.
// --shmSegment 로 런별 이름을 주면 동시 실행이 서로의 세그먼트에 붙지 않는다.
// INFRASTRUCTURE ONLY: 이름은 관측/행동/보상/구조/RNG 스트림/시드 매핑/시나리오/
// 타이밍/학습/패킷 회계/공유메모리 구조체 레이아웃에 아무 영향이 없다.
static std::string             g_shmSegment        = "TvtAiSeg";
static double                  g_todDataRateMbps   = 10.0;  // main()에서 todDataRateMbps로 갱신
static uint16_t                g_uuPortGlobal = 0;

// ---- [C3] RSU 릴레이 (SL 수신 → Uu 포워딩 → Server) ----
static std::vector<Ipv4Address>                  g_rsuBackhaulAddrs;   // RSU 백홀 IP 목록
static std::vector<Ptr<Socket>>                  g_rsuRelayTxSockets;  // RSU SL TX 소켓 목록
static std::unordered_map<uint32_t, Ptr<Socket>> g_rsuRelaySocketMap;  // nodeId → TX 소켓

// ---- [C4] ToD RTT 루프백 (왕복 지연 목표 검증) ----
static Ptr<Socket> g_todAckSocketUu;   // carNode→server ACK 송신 소켓 — 모든 RAT의 ACK이 이 소켓만 사용
static bool        g_disableTodAck = false; // A/B 실험: per-packet ToD ACK 무선 송신/계측 비활성화
// [제거됨] g_todAckSocketPc5 (PC5 경로 전용 ACK 소켓, IF1에 BindToNetDevice).
// ACK은 delay/loss 측정용 계측 신호이며 실제 데이터 전달 경로와 무관하게
// 항상 Uu(제어 채널 성격)로 전송함 — RSU가 PC5 역방향 릴레이를 지원하지
// 않는 구조적 제약(RSU의 SL 수신 콜백 비활성화, RSU 라우팅 테이블에
// 8.0.0.0/8↔1.0.0.0/8을 잇는 경로 부재) 및 측정 목적상 불필요하다는
// 판단에 따라 제거함. carNode에 IF1 바인딩 소켓으로 remoteHost(1.0.0.0/8)에
// 보내면 라우팅 테이블에 해당 경로가 없어 ERROR_NOROUTETOHOST로 즉시
// 실패했었다(원인 확인 완료).
static Ptr<Socket>             g_mecAckRxSocket;       // server ACK 수신 소켓
static uint16_t                g_ackPort   = 9999;     // ACK 포트
static std::deque<double>      g_recentRtts;
static const uint32_t          kRttWindowSize = 10;
static Ipv4Address             g_serverGlobalAddr;
static std::unordered_map<uint32_t, double> g_pendingAcks; // seq→송신시각(s)

// ── [ACK 기반 delay/loss 측정 — DDQN 전달용] ──────────────────────────────
// 기존 ToD ACK 파이프라인(SeqTsSizeHeader seq+타임스탬프 → SendTodAck 에코 →
// NotifyTodAckRx RTT 계산 → CheckPacketTimeout 타임아웃 감지)을 그대로
// 재사용하고, 그 결과만 DDQN(env->curr_delay_ms / env->curr_loss_rate)에
// 새로 연결한다. 타임아웃 기준값/윈도우 크기는 TBD, 실측 후 재조정 필요 —
// 우선 판단주기(g_dqnDecisionIntervalMs)의 10배로 임시 설정한다. 이는
// 기존 CheckPacketTimeout()의 고정 타임아웃(1.0s, 판단주기 기본값 100ms
// 기준 정확히 10배)과 동일한 시점이므로, 별도 타임아웃 시계를 두어 동일
// seq가 이중으로 카운트되는 위험 없이 기존 로직에 그대로 편승한다.
static const double   kAckLossTimeoutMultiplier = 10.0; // TBD: 판단주기의 10배로 임시 설정
static const uint32_t kAckLossWindowSize        = 50;   // TBD: 최근 50개 전송 기준 loss rate

static double           g_lastAckDelayMs = -1.0; // 최근 ACK RTT(ms) — curr_delay_ms, 데이터 없으면 -1
static std::deque<bool> g_ackOutcomeWindow;      // true=ACK 수신 성공, false=타임아웃(loss), 최근 N개

// ---- [C6] CP 메시지 유형 (3종, 크기 기반 분류용) ----
enum CpMsgType : uint8_t { CP_ACCIDENT=0, CP_OBSTACLE=1, CP_PEDESTRIAN=2, CP_NUM=3 };
static uint32_t g_rxCpType[CP_NUM] = {0, 0, 0};  // 타입별 RX 카운터

// ---- [C7] BSM (Basic Safety Message) 구조체 ----
// SAE J2735 BSM Core Data (27바이트, 패딩 없음)
#pragma pack(push, 1)
struct GpsBroadcast {
    uint32_t nodeId;       // 차량 노드 ID
    uint32_t seqNum;       // 시퀀스 번호
    uint64_t timestampMs;  // 타임스탬프 (ms)
    double   lat;          // 위도  (GPS, 36.5 + pos.x/111000)
    double   lon;          // 경도  (GPS, 127.0 + pos.y/91000)
    float    speed_mps;    // 속도  (m/s)
    float    heading_deg;  // 방향각 (도)
    float    altitude;     // 고도  (m)
};  // 총 41바이트
#pragma pack(pop)

// ---- [C7] BSM 송신 관련 전역 ----
static Ptr<Socket>  g_slSendSocket;       // UE1 SL 송신 소켓
static uint32_t     g_txCpPackets = 0;    // BSM TX 카운터

static std::set<uint32_t> activatedUes;
static std::unique_ptr<SQLiteOutput> g_db;
static UeRlcRxOutputStats g_rlcRxStats;
// [수정] g_psschTxStats(UeMacPsschTxOutputStats) 제거 — SetDb()로 DB 연결을
// 해주지 않은 채 Save()를 호출해 SIGSEGV 크래시 발생
// (NrSlUeMac::DoNrSlSlotIndication → NotifySlPsschScheduling → Save →
//  WriteCache → SQLiteOutput::SpinExec). 이 프로젝트는 SQLite DB를 실제로
// 활용하지 않으므로, DB 연결을 추가하는 대신 Save() 호출 자체와 이 객체를
// 제거하고 HARQ 재전송 카운팅만 남긴다.

// PHY 측정값 (RSRP/SINR 콜백에서 갱신)
static double   g_lastUuRssi      = -120.0;  // Uu RSRP (dBm)
static double   g_lastSinr        = -10.0;   // Uu SINR (dB) — CheckAndSwitchInterface 임계값 판단용
static double   g_lastUuRsrp     = -140.0;  // Uu RSRP (dBm), ReportRsrp 트레이스로 갱신

// ── Reward telemetry: D1(평상시)/D2(핸드오버 중) 평균 지연, 방향 구분 없음 ──
static std::deque<double> g_recentE2eDelaysMs;   // 최근 E2E 지연 샘플 (최근 윈도우 평균용, 상태 로그 표시용)
static const size_t       kE2eDelayWindowSize = 20;

// TVT reward는 ACK 경로가 아닌 순방향 application deadline cohort를 사용한다.
struct TvtDeadlineRecord
{
    double txTimeSec = 0.0;
    bool deliveredInTime = false;
    // Measured one-way delay of the FIRST valid arrival, -1 if never
    // delivered. Kept per record so a matured cohort can report the delay of
    // exactly the packets it originated, not of whatever happened to arrive
    // in some later measurement window.
    double deliveryDelayMs = -1.0;
    // Decision tick that transmitted this logical packet. This is the
    // attribution key: the deadline ripens ~250 ms later, but the outcome is
    // credited back here.
    uint64_t cohortSerial = 0;
    bool hasCohort = false;
};
static std::map<uint32_t, TvtDeadlineRecord> g_tvtDeadlinePending;
static std::deque<bool> g_tvtDeadlineOutcomeWindow;
static const size_t g_tvtDeadlineWindowSize = 200;
static const double g_tvtDeadlineMs = 250.0;
static double g_tvtLastRxTimeSec = -1.0;
static double g_tvtLastKpiSampleTimeSec = 0.0;
static double g_tvtUuThroughputBps = 0.0;
static double g_tvtPc5ThroughputBps = 0.0;

static double   g_d1Sum   = 0.0;   // 핸드오버 아닐 때(D1) 누적 지연 합
static uint64_t g_d1Count = 0;     // D1 누적 표본 수
static double   g_d2Sum   = 0.0;   // 핸드오버 진행 중(D2) 누적 지연 합
static uint64_t g_d2Count = 0;     // D2 누적 표본 수

static double   g_delaySumMs = 0.0;   // 전체 런 누적 E2E 지연 합 (최종 요약용)
static uint64_t g_delayCount = 0;     // 전체 런 누적 지연 샘플 수
static double   g_delayMaxMs = 0.0;   // 전체 런 최대 지연

// ============================================================================
// RNG CONTROL POLICY — common_random_numbers_v1
// ============================================================================
// PHASE C-2 의 L0/L1/L2 비교는 통제되지 않았다: numActiveVeh 를 바꾸면 ego 의
// uu_arrivals 까지 달라졌는데, active vehicle 은 SL-only 이고 Uu 디바이스가
// 없으므로 물리적 인과 경로가 없다.
//
// 원인은 ns-3 의 자동 stream 할당이다. RandomVariableStream 은 기본
// Stream=-1 로 생성되고, 그때 RngSeedManager::GetNextStreamIndex() 를
// 소비한다. UE 디바이스를 9대 더 설치하면 그 디바이스들의 fading/channel
// RandomVariableStream 이 전역 카운터를 더 밀어내고, 이후에 생성되는 모든
// stream(= ego 의 채널 객체 중 나중에 만들어지는 것들, CAM 위상, load jitter)
// 의 stream 번호가 통째로 이동한다. 즉 load level 이 exogenous realization 을
// 바꾼다.
//
// 해법은 두 가지를 같이 쓴다.
//
//  1) DOMAIN_A (radio/channel): 디바이스 그룹마다 고정 stream base 를
//     AssignStreams() 로 명시한다. active vehicle 을 몇 대 설치하든 gNB/ego/RSU
//     의 채널 stream 은 같은 번호를 받는다.
//
//  2) DOMAIN_B/C/D (scenario generation): CreateObjectWithAttributes 로
//     생성 시점에 Stream 을 지정한다. 이렇게 하면 SetStream(k>=0) 경로를 타서
//     전역 카운터를 **전혀 소비하지 않고**, 자동 할당 영역(앞쪽 2^63)과 겹치지
//     않는 결정론적 영역(뒤쪽 2^63 + k)을 쓴다.
//
// 여기에 더해 scenario 난수는 numActiveVeh 와 무관하게 **항상 고정 개수만큼**
// 뽑는다(kActiveVehMaxPopulation). 그래야 L1 의 앞 9대가 L2 의 앞 9대와
// 비트 단위로 같고, load jitter 가 부하 수준과 독립이 된다.
//
// 이것은 scenario semantics 변경이 아니라 실험 통제다. 차량 수, CAM rate,
// CAM 크기, gNB 위치, mobility window, reward, action 은 변경하지 않는다.
// ============================================================================
static const char* kRngControlPolicy = "common_random_numbers_v1";

// DOMAIN_A: radio/channel. 그룹별로 넉넉히 떨어뜨려 겹치지 않게 한다.
static const int64_t kStreamBaseGnb       = 10000;
static const int64_t kStreamBaseRsuSl     = 20000;
static const int64_t kStreamBaseEgoSl     = 30000;
static const int64_t kStreamBaseEgoUu     = 40000;
static const int64_t kStreamBaseActiveVeh = 50000;   // 대수에 따라 커지지만
                                                     // 위 그룹들을 침범하지 않는다
// DOMAIN_B/C/D: scenario generation. 자동 할당 영역과 완전히 분리된 번호.
static const int64_t kStreamVehX      = 1000000;  // 초기 x 위치
static const int64_t kStreamVehSpeed  = 1000001;  // 속도
static const int64_t kStreamVehLane   = 1000002;  // 차선 jitter
static const int64_t kStreamCamPhase  = 1100000;  // CAM 초기 위상
static const int64_t kStreamLoadJitter= 1200000;  // load 전이 jitter

// scenario 난수를 항상 이만큼 뽑는다. numActiveVeh 가 몇이든 draw 수가 같아야
// prefix 가 보존되고 뒤쪽 domain 이 밀리지 않는다. 현재 후보 최대치(18)보다
// 충분히 크게 잡되, 바꾸면 realization 이 달라지므로 고정값으로 둔다.
static const uint32_t kActiveVehMaxPopulation = 64;

// ── Active surrounding vehicle placement constants ─────────────────────────
// x 범위는 vehicle_mobility.tcl 의 ego 궤적 x 범위(5027.4..6172.3) 실측값이다.
// 속도 범위는 같은 trace 의 실측 구간(평균 7.2, 최대 14.7 m/s)이다. 차선 오프셋
// 3.5 m 는 표준 차선 폭이다. 어느 값도 특정 duplication 결과를 만들기 위해
// 선택되지 않았다 — ego 가 실제로 달리는 도로의 기하에서 가져왔다.
static const double kActiveVehXMin        = 5027.4;
static const double kActiveVehXMax        = 6172.3;
static const double kActiveVehLaneOffsetM = 3.5;
static const double kActiveVehLaneJitterM = 1.0;
static const double kActiveVehSpeedMin    = 5.0;
static const double kActiveVehSpeedMax    = 14.7;
static const double kActiveVehHeightM     = 1.5;
// CAM/BSM: ETSI EN 302 637-2 의 T_GenCamMin = 100 ms (= 10 Hz) 는 표준이 허용하는
// 생성률의 **상한**이며 "typical" 이 아니다. payload 350 B 는 ETSI 실측 범위
// 190..550 B 의 평균이다. 둘 다 PROVISIONAL 이다.
static const uint32_t kCamPayloadBytes    = 350;
static const double   kCamIntervalMs      = 100.0;
// ego ToD 와 같은 시각에 시작해 측정 구간 전체에서 경쟁하도록 한다.
static const double   kCamStartSec        = 1.5;
// CAM 송신 on/off 게이트 (T3 load schedule). 계측 전용 카운터와 함께.
static bool     g_camEnabled    = true;
// VALIDATION CONTROL ONLY (--camSilent, 기본 false). true 면 active vehicle 을
// 전부 설치하되 CAM 을 한 개도 보내지 않는다. "차량이 존재하는 효과"와
// "차량이 송신하는 효과"를 분리하기 위한 통제 도구이며 시나리오 파라미터가
// 아니다. 기본값에서는 아무 영향이 없다.
static bool     g_camSilent     = false;
static uint64_t g_camAttempted  = 0;   // 송신을 시도한 CAM 수
static uint64_t g_camTxOk       = 0;   // socket Send 가 성공한 수
static uint64_t g_camTxFail     = 0;   // socket Send 가 실패한 수
static uint64_t g_camSuppressed = 0;   // load-off 구간에서 건너뛴 수
// T3 load schedule: 공칭 전이 시각(s). service 전이(15 s, 25 s)와 일부러
// 어긋나게 둔다 — 둘이 정렬되면 Transformer 가 service requirement 에 반응한
// 것인지 congestion 에 반응한 것인지 구분할 수 없다. 이 배치는 한 episode
// 안에서 VIDEO x low / VIDEO x high / SAFETY x low / SAFETY x high 네 조합을
// 모두 방문한다. PROVISIONAL.
static const double kLoadTransitionsSec[] = {8.0, 15.0, 21.0, 25.0};
static const size_t kLoadTransitionCount  = 4;
static const double kLoadJitterSec        = 2.0;   // +/- , rngRun 에서 유도
// E-9 long-horizon support. The nominal transition list becomes settable so a
// 400 s episode can replicate the 28 s pattern with period 28 s. The default is
// kLoadTransitionsSec verbatim, so a run that omits --loadTransitionsSec draws
// exactly the same 4 values from kStreamLoadJitter and stays bit-identical to
// the frozen 28 s campaign.
static std::vector<double> g_loadNominalSec(
    kLoadTransitionsSec, kLoadTransitionsSec + kLoadTransitionCount);
static std::vector<double> g_loadTransitions;      // jitter 적용된 실제 시각

// ── Channel Busy Ratio (실측) ──────────────────────────────────────────────
// NrSlUeMac 이 S-RSSI 임계 -110 dBm, 100 ms window 로 계산하는 3GPP CBR 이다.
// proxy 가 아니라 직접 지표다. LOGGING ONLY — observation / reward / action 에
// OBSERVATION_CONTRACT_VERSION is 8: contract 8 adds the upper-tier
// mobility context c_t (ego_speed_mps, ego_rsu_distance_m). The ACTION
// contract and the lower radio state are unchanged.
static uint64_t g_cbrSamples = 0;
static double   g_cbrSum     = 0.0;
static double   g_cbrMin     = 1e9;
static double   g_cbrMax     = -1e9;
static double   g_cbrLast    = -1.0;
// ── Contention / resource-pressure PROXY (SL relay scheduler) ──────────────
// 직접 CBR 이 enableSensing=false 에서 발생하지 않으므로 대리 지표를 쓴다.
//
// 어느 노드를 재야 하는가: ToD 는 DL(server->ego)이고 PC5 경로는
// server -> RSU backhaul -> RSU -> SL -> ego 다. 즉 duplicate copy 를 SL 로
// 실제 송신하는 주체는 **RSU** 이고, surrounding vehicle 의 CAM 과 같은 pool
// 에서 경쟁하는 것도 RSU 의 grant 다. ego 는 SL 수신측이다.
//
// 이것은 CBR 이 아니다 — pool 점유율이 아니라 중계기가 자원을 얼마나 자주
// 요구하고 재선택하는지를 센다. LOGGING ONLY.
static uint64_t g_slTxGrants      = 0;
static uint64_t g_slTxReselect    = 0;
static uint64_t g_slTxTbPublished = 0;
static uint64_t g_slTxTbBytes     = 0;
static std::set<uint32_t> g_slTxGrantDst;
static void
SlTxGrantCreatedCallback(const NrSlUeMacScheduler::GrantInfo& grant,
                        uint16_t /*psfchPeriod*/)
{
    if (grant.slotAllocations.empty()) return;
    const uint32_t dst = grant.slotAllocations.begin()->dstL2Id;
    ++g_slTxGrants;
    // 이미 본 목적지에 다시 grant 가 나면 재선택이다 (기존 RSU 계측과 동일 기준).
    if (!g_slTxGrantDst.insert(dst).second) ++g_slTxReselect;
}
static void
SlTxGrantPublishedCallback(const NrSlUeMac::NrSlGrant& grant,
                          uint16_t /*psfchPeriod*/)
{
    ++g_slTxTbPublished;
    g_slTxTbBytes += grant.tbSize;
}

static void
EgoCbrCallback(double cbr)
{
    ++g_cbrSamples;
    g_cbrSum += cbr;
    g_cbrMin = std::min(g_cbrMin, cbr);
    g_cbrMax = std::max(g_cbrMax, cbr);
    g_cbrLast = cbr;
}


// ── LEGACY handover telemetry (default OFF) ────────────────────────────────
// 이 시뮬레이터는 물리 handover 를 수행하지 않는다. duplication degree 변경은
// 정수 하나를 바꾸는 것이고 bearer 재수립·시그널링·queue flush·interruption 이
// 없다. 아래 계측은 그런 handover 가 있던 시절의 산출물(Scenario1/2/3 CSV,
// D1/D2, tvt_ho_delay_*.csv, tvt_ho_times_*.csv)을 재현하기 위한 것이며,
// 측정된 D2-D1 차이는 0 이다.
//
// 과거 run 을 재현할 수 있도록 남겨두되 기본값은 false 다. false 일 때 이
// 계측은 state/reward/action 에 어떤 영향도 주지 않는다: 유일한 소비자가
// g_d1Sum/g_d2Sum 과 CSV 이고, 어느 쪽도 env 로 나가지 않는다.
// (g_switchCount 와 g_lastActionSwitchTimeSec 는 여기 포함되지 않는다 —
//  env 계약의 is_switch / time_since_last_switch_ms 를 만들기 때문이다.)
static bool   g_enableLegacyHoTelemetry = false;
static bool   g_interruptionActive     = false; // 지금 핸드오버 진행 중(감시구간) 여부
static double g_interruptionStartSimTime = -1.0;
static const double kInterruptionWindowSec = 0.05; // 전환 시작 후 50ms를 감시구간으로 간주

static uint32_t g_txTodPacketsWindow        = 0; // 10ms 윈도우 스냅샷 (loss 계산용)
static uint32_t g_rxTodPacketsWindowForLoss = 0;

static bool     g_sinrMeasured    = false;   // PHY 콜백으로 첫 측정값 수신 여부 (Python 전송 시작 가드)
static uint32_t g_ue1NodeId  = UINT32_MAX;

static uint32_t g_switchCount    = 0;
// Simulation time of the last applied action change. Diagnostic only: it
// gates nothing, because an action change costs the modelled network nothing.
static double   g_lastActionSwitchTimeSec = -2.0;

// ── DDQN 실행 모드 / 결정 주기 (CLI로 설정, main()에서 대입) ──
static std::string g_ratMode              = "ddqn";   // "ddqn" | "dup0".."dup100"
static double      g_dqnDecisionIntervalMs = 100.0;   // DDQN 결정 주기 (ms)

// ── ACTION HOLD: REMOVED FROM THE ARCHITECTURE ───────────────────────────
// The hold-semantics audit established that an action change here is traffic
// steering, not a handover: SetAction() assigns one integer and the next
// logical packet leaves on a different, already established socket. There is
// no attach/detach, no bearer reconfiguration, no route change, no socket
// recreation, no signalling, no queue flush, no interruption and no modelled
// switching latency, so no physical cost exists for a temporal hold to
// represent. Imposing one would inject a 500 ms handcrafted prior into an
// experiment whose subject is a LEARNED strategy, and it removed the action
// choice entirely in ~74% of decisions.
//
// RETIRED (LEGACY_ONLY): minRatHoldTimeMs, ratSwitchHoldMs,
// duplicationModeHoldMs. The CLI options are gone, so passing one is a hard
// command-line error rather than a silently ignored request.
static constexpr bool kActionHoldEnabled = false;
static bool        g_enablePc5Debug        = false;   // PHY/scheduler/link-budget 계측 on/off

// ── Observation warm-up ───────────────────────────────────────────────────
// Decisions (and therefore observations) start first so the planner's context
// history can be filled with REAL frames. Application traffic, formal reward
// and formal KPI accounting all begin together at g_trafficStartSec, so no
// zero-padded frame is ever a formal decision input and no pre-decision packet
// contaminates the reported statistics.
static uint32_t g_observationWarmupSteps = 10;   // = HISTORY_LEN
static double   g_decisionStartSec       = 2.0;  // first observation/decision
static double   g_trafficStartSec        = -1.0; // set in main(): decision + N*interval
static bool     g_formalAccountingActive = false;

// ── 선택/전환 telemetry (최종 요약 출력용) ─────────────────
// 요청된 duplication level 별 스텝 수. index == action index.
static uint32_t g_levelSelectedCount[TVT_AI_ACTION_COUNT] = {0, 0, 0, 0, 0};
static uint32_t g_invalidActionCount       = 0; // Python이 0/1 이외의 값을 반환한 횟수
static uint32_t g_switchCountAtLastDecision = 0; // 직전 결정 시점의 g_switchCount 스냅샷(is_switch 계산용)

// ── Time on Uu / Time on PC5 (누적, CheckAndSwitchInterface에서 갱신) ──
// duplication level 별 체류 시간. index == action index.
static double   g_timeOnLevelSec[TVT_AI_ACTION_COUNT] = {0.0, 0.0, 0.0, 0.0, 0.0};
// Forced Duplication Test 훅: -1 이면 비활성(ddqn 모드), 0..4 면 그 level 로 고정.
static int      g_forcedDuplicationLevel = -1;
static double   g_lastIfChangeAccountTime = 0.0;

// ── 차량-RSU 최소 거리 (전체 런 기준) ──────────────────────
static double   g_minRsuCarDist     = 1e18;
static double   g_minRsuCarDistTime = -1.0;

// ── 시나리오 측정 전역 변수 ────────────────────────────────────────────────
// 기본 출력 디렉토리: 이 소스 파일과 같은 디렉토리 (nr-v2x-examples/).
// --scenarioOutDir 가 주어지면 g_scenOutDir 가 그 값으로 대체된다.
static const std::string SCEN_OUT_DIR_DEFAULT = []() {
    std::string f = __FILE__;          // 컴파일 시각의 소스 전체 경로
    auto pos = f.rfind('/');
    return (pos != std::string::npos) ? f.substr(0, pos + 1) : std::string("./");
}();
static std::string g_scenOutDir = SCEN_OUT_DIR_DEFAULT;

static int    g_hoMeasIdx          = 0;     // 핸드오버 측정 인덱스
static bool   g_pendingHoMeas      = false; // 측정 대기 중 플래그
static double g_pend_s1_inferWall  = 0.0;  // S1: Python inference start (Unix wall-clock)
static double g_pend_s2_D1         = 0.0;  // S2: CppSendBegin 직전 시뮬 시각
// S3: CppSendBegin 직전 → switch_complete (ns3→Python IPC + DDQN + switching)
static double g_pend_s3_sendWall   = 0.0;  // S3 시작: CppSendBegin 직전 wall-clock
static double g_pend_s3_inferWall  = 0.0;  // S3 Delay1/2 경계: Python infer_start_wall_s
static double g_pend_s3_actionWall = 0.0;  // S3 Delay2/3 경계: Python action_sent_wall_s
static std::ofstream g_csvS1, g_csvS2, g_csvS3;

// ── Scenario 2: 핸드오버 구간 vs 평상시(정상 상태) 구간의 지연 비교 ──────
// "핸드오버 구간" = D1(결정)~D2(전환 감지) 사이의 시뮬레이션 시간 간격
//                    (Scenario1/3의 Interface Switching Delay와 동일 기준)
// "평상시 구간"   = targetInterface==currentInterface(유지) 결정 이후 다음
//                    100ms 폴링까지의 간격 — Modified: Action semantics changed
//                    (0=Uu, 1=PC5); action==currentAction 일 때만 "유지"로 판단
//                    (동일한 폴링 구조에서 실제로 아무 것도 바뀌지 않을 때의
//                     기본 반응 지연 — 핸드오버가 여기에 얼마나 추가 비용을
//                     더하는지 비교하기 위한 baseline)
static double   g_lastNonHoDecisionSimTime = -1.0; // 유지(action==currentAction) 결정 시각(시뮬 시간)
static double   g_curNonHoDelaySum         = 0.0;  // 직전 HO 종료~이번 HO 시작 구간 누적
static uint32_t g_curNonHoDelayCount       = 0;
static double   g_grandNonHoDelaySum       = 0.0;  // 전체 런 누적 (최종 요약용, 리셋 안 함)
static uint32_t g_grandNonHoDelayCount     = 0;
static double   g_grandHoDelaySum          = 0.0;
static uint32_t g_grandHoDelayCount        = 0;

// 측정 이상치 기준: 이 값을 초과하는 wall-clock 델레이는 CheckAndSwitchInterface()의
// 100ms 폴링 주기 사이에 다른 무관한 이벤트(Python IPC 왕복 등)의 실제 CPU 처리
// 시간이 함께 누적된 것으로 간주하고(호스트 스케줄링 잡음), 동일 구간의
// 시뮬레이션 시간 기준 값으로 대체한다 — 실제 핸드오버 로직/타이밍은 그대로 둔 채
// "측정값"만 보정한다.
static constexpr double kMaxPlausibleDelayMs = 250.0;

// POSIX CLOCK_REALTIME 기반 Unix time (Python time.time()과 동일 기준)
static inline double ScenarioWallNow() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

static uint64_t g_totalRxBitsUu = 0;
static uint32_t g_rxPacketsUu   = 0;

// ============================================================
// [C7] DDQN V2xState 구조체 및 전역 변수
// ============================================================

/**
 * \brief 무선 상태 스냅샷 (결정 주기마다 갱신)
 *
 * uu_rssi/pc5_rssi_avg는 각각 NrInterference::RssiPerProcessedChunk /
 * NrSlInterference::SlRssiPerProcessedChunk 실측값(UuRssiCallback/
 * Pc5RssiCallback)으로 갱신된다. 두 경로 모두 항상 측정된다 — 어느
 * duplication degree 에서도 secondary 경로의 상태를 알아야 "복제본을 하나 더
 * 보낼 가치가 있는가"를 판단할 수 있기 때문이다.
 * currentAction 은 현재 적용 중인 duplication degree (0=0%, 1=25%, 2=50%,
 * 3=75%, 4=100%)이며 경로 식별자가 아니다.
 */
struct V2xState
{
    double uu_rssi      = -120.0; ///< Uu RSSI (dBm), UuRssiCallback 갱신
    double pc5_rssi_avg = -100.0; ///< PC5 RSSI (dBm), Pc5RssiCallback 갱신
    int    currentAction =    0;  ///< 현재 duplication degree (0..4), 경로 아님
};

static V2xState g_dqnState;          ///< 현재 DDQN 상태

// PC5 채널 실측값 (RxPsschTraceUe 콜백에서 갱신)
static double g_slSinr_dB   = -10.0;
static double g_lastPc5Rssi = -100.0;   // PC5 RSRP (dBm) — -100→-85 중립값

// ============================================================================
// OPTIONAL RADIO DIAGNOSTICS  (--enablePc5Debug, default OFF)
//
// formal runtime core 와 분리된 계측이다. 아래 전역/콜백/1초 주기 printer 는
// g_enablePc5Debug 가 true 일 때만 트레이스에 연결되고 출력된다. state /
// reward / action / env 계약에 관여하지 않는다.
//
// 보존 사유: link budget, LOS/NLOS/NLOSv 판정, SL 스케줄러 grant, RLC 큐는
// PHASE C(multi-vehicle SCENARIO REDESIGN)에서 PC5 contention 과 interference
// 를 검증할 때 그대로 필요하다. 예전 태그가 [TEMP TEST] 였으나 임시 코드가
// 아니므로 [RADIO-DIAG] 로 바꿨다 — 기본 실행의 stdout 에는 나타나지 않는다.
// ============================================================================
// ── [RADIO-DIAG] PC5 SL MAC 스케줄러 계측 (reservationPeriod 병목 정량 검증용) ──
// RSU의 NrSlUeMacScheduler에 연결된 GrantCreated/GrantPublished/RlcBufferStatus
// 트레이스를 1초 윈도우로 집계한다. 스케줄러/RLC 동작 자체는 변경하지 않음.
static uint32_t g_pc5GrantsThisSec       = 0;  // 이번 1초 동안 생성/재선택된 grant 수
static uint32_t g_pc5ReselectionsThisSec = 0;  // 그중 "재선택"(기존 목적지 재사용)으로 판단된 수
static std::set<uint32_t> g_pc5SeenDst;        // 목적지별 최초 생성 여부 판별용
static uint32_t g_pc5TbPublishedThisSec  = 0;  // 이번 1초 동안 MAC에 실제 전달된 grant(TB) 수
static uint64_t g_pc5TbBytesSumThisSec   = 0;  // TB 크기 합 (평균 계산용)
static uint8_t  g_pc5LastNSelected       = 0;  // 마지막 grant의 예약 슬롯 수(nSelected, HARQ 포함)
static double   g_pc5LastRriMs           = -1.0; // 마지막으로 관측된 실제 RRI(ms)
static uint32_t g_pc5LastRlcQueueBytes   = 0;  // 마지막 RLC 신규 송신 큐 크기(bytes)
static uint32_t g_pc5LastRetxQueueBytes  = 0;  // 마지막 RLC 재전송 큐 크기(bytes)

// ── [RADIO-DIAG] PC5 링크버짓 계측 (SINR 저하 원인 분리용) ──────────────
// carNode의 NrSlInterference에 연결된 SlLinkBudget 트레이스(신호/간섭/잡음/SINR
// 분리값)를 누적한다. 계산 로직/파라미터는 변경하지 않고 값만 관찰한다.
static double   g_pc5LastSignalDbm  = NAN;
static double   g_pc5LastInterfDbm  = NAN;
static double   g_pc5LastNoiseDbm   = NAN;
static double   g_pc5LastSinrDb     = NAN;
static uint32_t g_pc5LinkBudgetSamplesThisSec = 0;
static double   g_pc5SinrSumThisSec = 0.0;

// [RADIO-DIAG] A/B/C 실험 전체 런 누적치 (평균/최소/최대 계산용)
static uint64_t g_pc5RunSampleCount   = 0;
static double   g_pc5RunSignalSum     = 0.0;
static double   g_pc5RunSignalMin     = 1e9;
static double   g_pc5RunSignalMax     = -1e9;
static double   g_pc5RunSinrSum       = 0.0;
static double   g_pc5RunSinrMin       = 1e9;
static double   g_pc5RunSinrMax       = -1e9;

// [RADIO-DIAG] ChannelConditionModel의 LOS/NLOS/NLOSv 판정 횟수 (전체 런 누적)
static uint64_t g_losCount   = 0;
static uint64_t g_nlosCount  = 0;
static uint64_t g_nlosvCount = 0;

static uint64_t g_uuLosCount   = 0;
static uint64_t g_uuNlosCount  = 0;
static uint64_t g_uuNlosvCount = 0;

// 패킷별 E2E 지연 기록. 핸드오버 구간/비핸드오버 구간을 사후에 나누려면
// 수신 시각이 필요하므로 rxTimeSec를 함께 남긴다.
struct PacketDelayRecord
{
    uint32_t seq;
    double   delayMs;
    double   rxTimeSec;
};
static std::vector<PacketDelayRecord> g_packetDelayLog;
static std::vector<std::pair<uint32_t, double>> g_hoDelayLog;
// 실제 전환이 일어난 시뮬레이션 시각과 방향 — 패킷을 핸드오버 구간/비핸드오버
// 구간으로 사후 분류하는 데 쓴다.
static std::vector<std::pair<double, std::string>> g_hoSwitchTimes;
static uint32_t g_hoEventIndex = 0;

// [RADIO-DIAG] PSCCH(좁은 대역, occupiedRbCount<=15)/PSSCH(넓은 대역) 분리 집계
// — SlLinkBudget이 두 채널타입을 섞어서 min/max를 왜곡하는지 검증용.
static const uint32_t kNarrowRbThreshold = 15;
static uint64_t g_narrowSampleCount = 0;
static double   g_narrowSignalSum   = 0.0;
static double   g_narrowSignalMin   = 1e9;
static double   g_narrowSignalMax   = -1e9;
static uint64_t g_wideSampleCount   = 0;
static double   g_wideSignalSum     = 0.0;
static double   g_wideSignalMin     = 1e9;
static double   g_wideSignalMax     = -1e9;

// [RADIO-DIAG] NLOSv 추가손실(GetAdditionalNlosvLoss) / 일반 Shadowing 손실 — 별도 집계
static uint64_t g_nlosvLossCount = 0;
static double   g_nlosvLossSum   = 0.0;
static double   g_nlosvLossMin   = 1e9;
static double   g_nlosvLossMax   = -1e9;
static uint64_t g_shadowingCount = 0;
static double   g_shadowingSum   = 0.0;
static double   g_shadowingMin   = 1e9;
static double   g_shadowingMax   = -1e9;

// 손실률·스루풋 계산용 윈도우 스냅샷 (CollectV2xState 호출마다 갱신)
static uint64_t g_uuBytesWindow    = 0; ///< CollectV2xState 윈도우 (경로 관측용)
static uint64_t g_slBytesWindow    = 0; ///< CollectV2xState 윈도우 (경로 관측용)
// ── AUTHORITATIVE useful goodput ─────────────────────────────────────────
// 위 두 카운터는 duplicate 판정 **이전** 에 누적되는 경로별 byte 수다. 즉
// duplicate 도 포함하고, 경로마다 따로 센다. 어느 쪽도 application goodput 이
// 아니다: logical packet 은 두 경로 중 먼저 도착한 쪽이 전달하므로 실제 양은
// 경로에 대한 합집합이고, 최댓값은 합집합을 표현할 수 없다.
//
// 아래 카운터는 first-arrival 분기에서만, 선언된 payload byte 로만 증가한다.
// 같은 seq 는 copy 가 몇 개 오든 정확히 한 번. probe/헤더 byte 는 제외.
static uint64_t g_logicalGoodputBytesWindow = 0;
static double   g_tvtLogicalGoodputBps      = 0.0;
static uint64_t g_logicalGoodputBytesTotal  = 0;   ///< 독립 검산용 누적

// ── 스루풋 지수 이동 평균 (EMA, α=0.3) ──────────────────────────────────
// 10ms 순간 윈도우가 0이 돼도 이전 값을 보존 → R_thpt 안정화.
// 초기값 0, 첫 패킷 수신 후 빠르게 실제값으로 수렴.
static double g_uuThptEma = 0.0;   ///< Uu 스루풋 EMA (kbps)
static double g_slThptEma = 0.0;   ///< PC5 스루풋 EMA (kbps)
static double g_emaUuThpt  = 0.0;  ///< Uu ToD 수신 패킷 기반 EMA (bps)
static double g_emaSlThpt  = 0.0;  ///< PC5 ToD 수신 패킷 기반 EMA (bps)
static const double kAlpha = 0.1;  ///< 패킷 수신 EMA 평활 계수
static uint64_t g_uuBytesPrintWin  = 0; ///< PrintThroughput 1s 윈도우 (로그 표시 전용)
static uint64_t g_slBytesPrintWin  = 0; ///< PrintThroughput 1s 윈도우 (로그 표시 전용)

// 스위칭 빈도 추적 (최근 10s 핸드오버 타임스탬프 ring)
static std::deque<double> g_hoTimestamps;

// 차량 노드 포인터 + 전체 UE 컨테이너 (main에서 설정)
static Ptr<Node>      g_carNodePtr     = nullptr;
static NodeContainer  g_allUesContainer;           ///< 차량 밀집도 계산용
static uint32_t       g_activePc5RsuIdx = 0;       ///< 현재 활성 PC5 RSU 인덱스
static NodeContainer  g_rsuNodesGlobal;            ///< RSU 전용 NodeContainer

// ── 5G NR-V2X HARQ 재전송 추적 (NDI=0 PSSCH 슬롯 카운트) ──
// NotifySlPsschScheduling() 콜백에서 ndi==0 검출 시 증가.
// 100ms 윈도우 차분으로 harqReTxRate 계산.
static uint32_t g_slHarqReTxCount = 0;     ///< 누적 SL 재전송 횟수
static uint32_t g_slHarqTxCount   = 0;     ///< 누적 SL 전체 전송 횟수

// ── [C8] ns3-ai 공유 메모리 인터페이스 ─────────────────────────────────────
// Python (tvt_test.py) 이 먼저 공유 메모리를 생성(creator=true)한 뒤
// ns-3 가 open_only 로 접속한다. 접속 실패 시 nullptr → requestedAction=0(Uu 목표) 유지.
static ns3::Ns3AiMsgInterfaceImpl<ns3::TvtAiEnv, ns3::TvtAiAct>* g_aiInterface = nullptr;

struct AiActionResult
{
    uint64_t decisionId = 0;
    int requested = 0;
    int applied = 0;
    TvtAiApplyStatus status = TVT_AI_APPLY_NONE;
    bool valid = false;
};

static uint64_t g_nextAiDecisionId = 1;
static AiActionResult g_lastAiResult;
static bool g_pendingAiDecision = false;
static uint64_t g_pendingAiDecisionId = 0;
static int g_pendingAiRequestedAction = 0;
static TvtAiEnv g_lastSentAiEnv{};
static bool g_hasLastSentAiEnv = false;

static const char*
AiApplyStatusName(TvtAiApplyStatus status)
{
    switch (status)
    {
    case TVT_AI_APPLY_STAY: return "STAY";
    case TVT_AI_APPLY_APPLIED: return "APPLIED";
    // RETIRED (LEGACY_ONLY): the hold that produced this status is gone.
    case TVT_AI_APPLY_HOLD_VETO_RETIRED: return "HOLD_VETO_RETIRED";
    case TVT_AI_APPLY_INVALID_ACTION: return "INVALID_ACTION";
    case TVT_AI_APPLY_DECISION_ID_MISMATCH: return "DECISION_ID_MISMATCH";
    case TVT_AI_APPLY_PENDING: return "PENDING";
    case TVT_AI_APPLY_CONTRACT_MISMATCH: return "CONTRACT_MISMATCH";
    default: return "NONE";
    }
}

// ============================================================
// [B] Path transmission + per-path measurement (MP-QUIC removed)
//
// The Uu and PC5 paths are two independent, permanently configured UDP
// sockets on the server. Neither is ever created or torn down at decision
// time: an action selects HOW MANY copies of the next logical application
// packets go out, it never attaches or detaches connectivity. Raising or
// lowering the duplication degree therefore costs nothing beyond the extra
// physical copies it transmits.
// ============================================================

// ── [B0] Generic measurement header (2 B) ─────────────────────────
// This is NOT a QUIC header. There is no connection ID, no packet number, no
// handshake and no frame taxonomy. It exists only so that the shared
// ACK/probe UDP port can tell an ACK from a probe response and attribute it
// to the path that produced it.
enum TvtMeasType : uint8_t {
    TVT_MEAS_ACK        = 1,   // carNode -> server: logical packet seq echo
    TVT_MEAS_PROBE      = 2,   // server  -> carNode: liveness / RTT probe
    TVT_MEAS_PROBE_RESP = 3    // carNode -> server: probe echo
};

class TvtMeasHeader : public Header {
public:
    uint8_t type   = TVT_MEAS_ACK;
    uint8_t pathId = 0;        // 0 = Uu, 1 = PC5

    static TypeId GetTypeId() {
        static TypeId tid = TypeId("ns3::TvtMeasHeader")
            .SetParent<Header>()
            .SetGroupName("Applications")
            .AddConstructor<TvtMeasHeader>();
        return tid;
    }
    TypeId GetInstanceTypeId() const override { return GetTypeId(); }
    uint32_t GetSerializedSize() const override { return 2; }
    void Serialize(Buffer::Iterator s) const override {
        s.WriteU8(type);
        s.WriteU8(pathId);
    }
    uint32_t Deserialize(Buffer::Iterator s) override {
        type   = s.ReadU8();
        pathId = s.ReadU8();
        return 2;
    }
    void Print(std::ostream& os) const override {
        os << "TvtMeasHeader type=" << (uint32_t)type
           << " pathId=" << (uint32_t)pathId;
    }
};

// ── [B1] Per-path measurement ─────────────────────────────────────
// Kept fresh for BOTH paths on every decision interval regardless of which
// action is in force. Every value comes from a real event: a data arrival on
// that path, or the 100 ms probe when that path is carrying no data. Nothing
// here is synthesised, extrapolated or copied from the other path.
struct PathMeasurement {
    double rssiDbm       = -120.0;
    double delayMs       = -1.0;   // EMA of one-way delay; -1 = never measured
    double thptBps       = 0.0;
    double lastUpdateSec = -1.0;
    bool   valid         = false;

    static constexpr double kDelayAlpha = 0.3;

    void NoteDelaySample(double sampleMs, double nowSec) {
        if (!std::isfinite(sampleMs) || sampleMs < 0.0) return;
        delayMs = (delayMs < 0.0)
                    ? sampleMs
                    : kDelayAlpha * sampleMs + (1.0 - kDelayAlpha) * delayMs;
        lastUpdateSec = nowSec;
        valid = true;
    }
    double AgeMs(double nowSec) const {
        return (lastUpdateSec < 0.0) ? 1e9 : (nowSec - lastUpdateSec) * 1000.0;
    }
};

// ── [B2] Per-path loss accounting ─────────────────────────────────
// Sliding cohort of completed probe outcomes for one path: true = the probe
// was echoed back, false = it timed out. This replaces the loss bookkeeping
// that used to live inside the MP-QUIC reorder buffer, without any of its
// packet-number or reassembly semantics.
struct PathLossAccounting {
    static constexpr size_t kWindow = 50;
    std::deque<bool> outcomes;

    void Note(bool delivered) {
        outcomes.push_back(delivered);
        if (outcomes.size() > kWindow) outcomes.pop_front();
    }
    bool HasSamples() const { return !outcomes.empty(); }
    double Rate() const {
        if (outcomes.empty()) return 0.0;
        size_t lost = 0;
        for (bool ok : outcomes) if (!ok) ++lost;
        return static_cast<double>(lost) / static_cast<double>(outcomes.size());
    }
};

// ── [B3] Redundancy transmit manager ──────────────────────────────
// Sends one logical application packet as 1 or 2 physical copies according to
// the DUPLICATION DEGREE in force. Path identity is fixed policy:
//
//     PRIMARY   (Uu)  -- every logical packet, always
//     SECONDARY (PC5) -- the fraction of packets the accumulator selects
//
// Both copies carry the SAME application sequence number and timestamp, so the
// receiver recognises the second one as a duplicate of the same logical packet.
//
// WHICH packets get the second copy is decided by a DETERMINISTIC CREDIT
// ACCUMULATOR, never by a random draw:
//
//     credit += ratio
//     primary copy always attempted
//     if credit >= 1:  secondary copy attempted;  credit -= 1
//
// Two properties follow, and both are the reason this is not a Bernoulli
// sampler. First, the ns-3 RNG stream is untouched, so changing the
// duplication degree cannot perturb the channel or mobility realisation and
// the run stays bitwise reproducible. Second, the accumulated count never
// drifts: after any number of packets at any sequence of ratios, the number of
// duplicates differs from the sum of the ratios by strictly less than one
// packet, because that difference IS the retained credit, which is held in
// [0, 1) by construction.
class RedundancyTxManager {
public:
    struct PathEntry {
        Ptr<Socket> sock;
        Address     dst;
        bool        valid = false;
        uint32_t    txCopies = 0;
        uint64_t    txBytes  = 0;
    };

    PathEntry m_paths[2];
    int       m_action = TVT_AI_ACTION_DUP_0;
    double    m_credit = 0.0;      // in [0, 1) between packets
    // Diagnostics for the accumulator invariant test. Not used by any reward.
    double    m_ratioSum = 0.0;    // sum of the ratio charged per packet
    uint64_t  m_duplicated = 0;    // packets that actually got a second copy
    uint64_t  m_eligible = 0;      // packets the accumulator was consulted for

    void AddPath(int id, Ptr<Socket> sock, const Address& dst) {
        if (id < 0 || id > 1) return;
        m_paths[id].sock  = sock;
        m_paths[id].dst   = dst;
        m_paths[id].valid = true;
    }

    /** Set the duplication degree. Changing it never resets the credit: the
     *  accumulator carries across the change, which is what bounds the error
     *  over a ratio SEQUENCE and not merely within one constant stretch. */
    void SetAction(int action) {
        if (action < 0 || action >= TVT_AI_ACTION_COUNT) return;
        m_action = action;
    }
    int GetAction() const { return m_action; }
    double GetRatio() const { return TvtAiDuplicationRatio(m_action); }

    /** Reset per episode. The credit must not leak across episodes, or the
     *  first packets of an episode would depend on the tail of the previous
     *  one and the episodes would stop being independent. */
    void ResetAccumulator() {
        m_credit = 0.0;
        m_ratioSum = 0.0;
        m_duplicated = 0;
        m_eligible = 0;
    }

    int PrimaryPath() const { return TVT_AI_PRIMARY_PATH; }
    int SecondaryPath() const { return TVT_AI_SECONDARY_PATH; }

    /** Transmit one physical copy on one path. Returns true if it went out. */
    bool SendCopy(int pathId, Ptr<Packet> copy) {
        if (pathId < 0 || pathId > 1) return false;
        auto& p = m_paths[pathId];
        if (!p.valid || !p.sock) return false;
        if (p.sock->SendTo(copy, 0, p.dst) <= 0) return false;
        p.txCopies++;
        p.txBytes += copy->GetSize();
        return true;
    }

    /** Advance the accumulator for one logical packet and say whether that
     *  packet is duplicated. Pure integer/float arithmetic, no RNG. */
    bool ConsumeDuplicationCredit() {
        const double ratio = GetRatio();
        ++m_eligible;
        if (ratio <= 0.0) return false;
        m_ratioSum += ratio;
        m_credit += ratio;
        if (m_credit >= 1.0 - kCreditEpsilon) {
            m_credit -= 1.0;
            if (m_credit < 0.0) m_credit = 0.0;   // guard the 1.0 case exactly
            ++m_duplicated;
            return true;
        }
        return false;
    }

    /** Copies the degree in force calls for on the NEXT logical packet. This
     *  is a look-ahead for telemetry only; it does not advance the credit. */
    int IntendedCopies() const {
        const double ratio = GetRatio();
        if (ratio <= 0.0) return 1;
        return (m_credit + ratio >= 1.0 - kCreditEpsilon) ? 2 : 1;
    }

    /** Send one logical packet. Returns the number of physical copies that a
     *  socket accepted; *attempted receives the number the degree called for.
     *  Redundant duplication of ONE logical packet, never a split, a stripe,
     *  an alternation or a path switch. */
    int SendLogicalPacket(Ptr<Packet> payload, int* attempted,
                          bool* duplicated = nullptr) {
        const bool second = ConsumeDuplicationCredit();
        if (duplicated) *duplicated = second;

        int successful = 0;
        int tried = 1;
        if (SendCopy(PrimaryPath(), payload->Copy())) ++successful;
        if (second) {
            ++tried;
            if (SendCopy(SecondaryPath(), payload->Copy())) ++successful;
        }
        if (attempted) *attempted = tried;
        return successful;
    }

private:
    // 0.25 and 0.75 are exact in binary; 0.5 and 1.0 trivially so. The epsilon
    // exists only so a ratio that sums to exactly 1.0 through repeated
    // addition is not lost to a last-bit rounding artefact.
    static constexpr double kCreditEpsilon = 1e-12;
};

static RedundancyTxManager g_pathTx;
static PathMeasurement   g_pathMeas[2];
static PathLossAccounting g_pathLoss[2];

// ── [B4] Logical packet accounting (duplicate-suppressed) ─────────
// One logical application packet == one application sequence number, however
// many physical copies were transmitted for it. Duplicates are counted as
// resource usage but must never inflate delivery, goodput or deadline
// success.
static uint64_t g_logicalTxPackets        = 0;
static uint64_t g_logicalRxPackets        = 0;   // first valid arrival only
static uint64_t g_logicalDeadlineSuccess  = 0;
static uint64_t g_logicalDeadlineMiss     = 0;
// attempted = copies the duplication degree in force called for (1 when the
//   accumulator does not select the packet, 2 when it does). Counted even when
//   a socket send then fails, so a duplicated packet can never be charged as
//   cheaply as a single copy. This is what the resource cost component reads.
// successful = copies a socket actually accepted. Diagnostics only.
static uint64_t g_attemptedCopiesTx       = 0;
static uint64_t g_successfulCopiesTx      = 0;
static uint64_t g_duplicateCopiesRx       = 0;
static uint64_t g_duplicateBytesRx        = 0;
// logical packet 중 실제로 두 번째 copy 를 받은 개수. 선택된 action 의 횟수가
// 아니라 중복 전송된 packet 수다 — partial duplication 에서 의미를 갖는 유일한
// 형태다.
static uint64_t g_duplicatedLogicalPackets = 0;
static uint64_t g_windowLogicalTx         = 0;   // reset every decision
static uint64_t g_windowAttemptedCopies   = 0;   // reset every decision
static uint64_t g_windowSuccessfulCopies  = 0;   // reset every decision
static uint64_t g_windowDuplicatedLogicalPackets = 0;   // reset every decision
// A logical packet's IDENTITY must outlive its deadline. Two sets, on purpose:
//   g_logicalSeen        every seq that has ever been delivered once. NEVER
//                        erased during an episode, so a copy arriving after
//                        the deadline matured can never masquerade as a first
//                        delivery. This is the invariant that keeps duplication
//                        from inflating logical goodput.
//   g_tvtDeadlinePending seqs still awaiting deadline maturation. Erased on
//                        maturation; that is a deadline lifetime, not identity.
static std::set<uint32_t> g_logicalSeen;
static uint64_t g_lateDuplicateRx = 0;   // copy arriving after maturation
static uint64_t g_lateDuplicateBytes = 0;

// ── Macro tick cohorts: reliability attribution ───────────────────────
// A deadline outcome ripens ~250 ms after transmission, so crediting it to
// whatever macro interval happens to be open when it ripens attributes an
// outcome to a goal that did not cause it. Instead every decision tick opens
// a COHORT tagged with the macro interval that transmits it; the outcome of
// each logical packet is credited back to its originating cohort, and the
// cohort is published to Python only once every one of its packets has
// reached its deadline.
//
// Throughput and outage are snapshotted when the cohort CLOSES, i.e. over
// exactly the 100 ms window the cohort's own packets were transmitted in, so
// all four KPIs of a cohort describe the same decision.
struct TvtTickCohort
{
    uint64_t serial = 0;
    uint64_t macroId = TVT_AI_NO_MACRO;   // NO_MACRO outside a goal interval
    int32_t  macroTick = -1;              // 0..H-1 within that macro
    int32_t  action = TVT_AI_ACTION_DUP_0; // duplication level in force
    double   openTimeSec = 0.0;
    double   closeTimeSec = -1.0;
    bool     closed = false;
    uint64_t logicalTx = 0;
    uint64_t pendingDeadlines = 0;
    uint64_t success = 0;
    uint64_t miss = 0;
    double   delaySumMs = 0.0;
    uint64_t delayCount = 0;
    uint64_t attemptedCopies = 0;
    uint64_t duplicatedLogicalPackets = 0;
    double   uuThptBps = 0.0;        // path-condition observation
    double   pc5ThptBps = 0.0;       // path-condition observation
    double   logicalGoodputBps = 0.0; // the cohort's USEFUL throughput
    double   timeSinceLastRxMs = 0.0;

    bool Complete() const { return closed && pendingDeadlines == 0; }
};
static std::map<uint64_t, TvtTickCohort> g_tickCohorts;   // serial -> cohort
static uint64_t g_nextCohortSerial   = 0;
static bool     g_haveOpenCohort     = false;
static uint64_t g_openCohortSerial   = 0;
static uint64_t g_cohortsOpened      = 0;
static uint64_t g_cohortsPublished   = 0;
static uint64_t g_cohortsDropped     = 0;   // evicted unpublished (guard only)
static const size_t kMaxLiveCohorts  = 512; // ~51 s of backlog; never reached

// Goal macro currently in force, as declared by Python in the act struct.
static uint64_t g_currentMacroId   = TVT_AI_NO_MACRO;
static int32_t  g_currentMacroTick = -1;

// Probes in flight: (pathId, probeSeq) -> send time.
static std::map<std::pair<int, uint32_t>, double> g_pendingProbes;
static uint32_t g_probeSeqNum = 0;
static const double kProbeTimeoutSec = 1.0;   // 10 x the 100 ms probe period

// ============================================================
// [F] 전방 선언
// ============================================================
void NotifySlRx(Ptr<Socket> socket);
void RxTraceCallback(ns3::Ptr<const ns3::Packet> packet);
Ptr<NrUeNetDevice> GetUeNetDevice(Ptr<Node> node);
void CheckAndSwitchInterface();
void UpdatePc5Path();
uint32_t FindNearestRsu(NodeContainer& rsuNodes, Ptr<Node> carNode);
// MOBILITY CONTEXT c_t (observation contract 8). Defined at [I-4b]; declared
// here because the terminal handshake env is filled earlier in the file.
static void FillMobilityContext(TvtAiEnv* env);
void PrintThroughput();
void PrintPc5SchedulerStats();  // [RADIO-DIAG] PC5 SL 스케줄러 계측 (원인 검증용)
void PrintPc5LinkBudgetStats(); // [RADIO-DIAG] PC5 링크버짓 계측 (SINR 원인 분리용)
void SendTodPeriodic(Time interval);
void NotifyRsuRelayRx(Ptr<Socket> socket);
void NotifyTodAckRx(Ptr<Socket> socket);     // MEC: ACK 수신 → RTT 계산
void NotifySlRxSizeDispatch(Ptr<Socket> socket);                 // 크기 기반 타입 자동 분류
void NotifyUe1PcToD(Ptr<Socket> socket);                        // UE1 PC5 릴레이 ToD 수신
void SendTodAck(uint32_t seq, Ipv4Address dst, uint8_t pathId);  // 논리 패킷 수신 에코
void SendSlPacket();                                             // BSM 100ms 주기 송신
void SendBsmForNode(Ptr<Node> node, Ptr<Socket> sock);           // 차량별 BSM 100ms (현재 미사용)
void ProbeBothPaths();                                           // 두 경로 100ms 주기 측정 프로브
void CheckPacketTimeout();                                       // ACK/프로브 타임아웃 감지
void SendMeasProbeResponse(uint8_t pathId, uint32_t probeSeq, Ipv4Address dst);

// ============================================================
// [E] 유틸리티: NrUeNetDevice 획득
// ============================================================
Ptr<NrUeNetDevice> GetUeNetDevice(Ptr<Node> node)
{
    for (uint32_t i = 0; i < node->GetNDevices(); ++i) {
        Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(node->GetDevice(i));
        if (ueDev) return ueDev;
    }
    return nullptr;
}

// ============================================================
// [F] SL RLC 수신 콜백 (DB 저장 + CP PDR 집계)
// ============================================================
void NotifySlRx(Ptr<Socket> socket)
{
    Address    from;
    Ptr<Packet> packet;
    while ((packet = socket->RecvFrom(from))) {
        std::string fromStr = "UNKNOWN";
        if (InetSocketAddress::IsMatchingType(from)) {
            Ipv4Address srcAddr = InetSocketAddress::ConvertFrom(from).GetIpv4();
            std::ostringstream oss;
            srcAddr.Print(oss);
            fromStr = oss.str();
        }
        g_rxCpPackets++;
        g_rxPacketsSl++;
        g_totalRxBitsSl += packet->GetSize() * 8;
    }
}


// ============================================================
// [K-BSM] BSM (Basic Safety Message) 100ms 주기 송신
// ------------------------------------------------------------
// 분류: INACTIVE_BUT_USEFUL_FOR_MULTI_VEHICLE  (삭제하지 말 것)
//
// 현재 formal 시나리오(ego 1 / activeVeh 0 / background 0)에서는 두 함수 모두
// 호출되지 않는다: g_slSendSocket 은 항상 nullptr 이고, SendBsmForNode 를
// 도는 루프는 numActiveVeh == 0 이라 0회 실행된다. 따라서 현재 측정·reward·
// state 에 어떤 영향도 주지 않는다.
//
// 그러나 이것은 obsolete 코드가 아니라 multi-vehicle 시나리오의 트래픽
// 생성기다. PHASE C(SCENARIO REDESIGN)에서 active surrounding vehicle 의
// V2X 트래픽·PC5 contention 을 만들 때 그대로 재사용한다. 그때까지 유지한다.
// ============================================================
void SendSlPacket()
{
    if (!g_slSendSocket) return;

    GpsBroadcast gps = {};
    gps.nodeId      = g_carNodePtr->GetId();
    gps.seqNum      = g_txCpPackets;
    gps.timestampMs = static_cast<uint64_t>(
        Simulator::Now().GetMilliSeconds());

    if (g_carNodePtr) {
        Ptr<MobilityModel> mob = g_carNodePtr->GetObject<MobilityModel>();
        Vector pos = mob->GetPosition();
        Vector vel = mob->GetVelocity();

        const double baseLat = 49.6117;
        const double baseLon = 6.1319;
        gps.lat      = baseLat + pos.x / 111000.0;
        gps.lon      = baseLon + pos.y / 91000.0;
        gps.altitude = static_cast<float>(pos.z);

        double spd = std::sqrt(vel.x*vel.x + vel.y*vel.y);
        gps.speed_mps = static_cast<float>(spd);

        double hdg = std::atan2(vel.y, vel.x) * 180.0 / M_PI;
        if (hdg < 0) hdg += 360.0;
        gps.heading_deg = static_cast<float>(hdg);
    }

    Ptr<Packet> pkt = Create<Packet>(
        reinterpret_cast<const uint8_t*>(&gps), sizeof(GpsBroadcast));

    g_slSendSocket->Send(pkt);
    g_txCpPackets++;

    NS_LOG_DEBUG("[GPS-Tx] lat=" << gps.lat << " lon=" << gps.lon
                 << " speed=" << gps.speed_mps << "m/s"
                 << " @ " << Simulator::Now().GetSeconds() << "s");

    double now = Simulator::Now().GetSeconds();
    if (now < g_simEndTime - 0.1)
        Simulator::Schedule(MilliSeconds(100), &SendSlPacket);
}

// Active surrounding vehicle 의 CAM/BSM 주기 송신.
//
// payload 는 kCamPayloadBytes(350 B, ETSI 실측 평균)로 맞춘다. GpsBroadcast 가
// 그보다 작으므로 나머지는 패딩이다 — 내용이 아니라 **PC5 자원 점유**가
// 목적이기 때문이다. 주기는 kCamIntervalMs(100 ms = ETSI T_GenCamMin, 표준
// 상한).
//
// g_camEnabled 가 false 인 구간에서는 송신을 건너뛰되 스케줄은 유지한다.
// 이것이 T3(load temporal variation)의 구현이다: 노드를 생성/삭제하는 것이
// 아니라 송신을 켜고 끈다 — topology 가 아니라 network load 를 바꾼다.
void SendBsmForNode(Ptr<Node> node, Ptr<Socket> sock)
{
    double now = Simulator::Now().GetSeconds();
    if (now >= g_simEndTime - 0.1) return;

    if (g_camSilent || !g_camEnabled)
    {
        ++g_camSuppressed;
        Simulator::Schedule(MilliSeconds(static_cast<uint64_t>(kCamIntervalMs)),
                            &SendBsmForNode, node, sock);
        return;
    }

    GpsBroadcast gps = {};
    gps.nodeId      = node->GetId();
    gps.seqNum      = 0;
    gps.timestampMs = static_cast<uint64_t>(Simulator::Now().GetMilliSeconds());

    Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
    if (mob) {
        Vector pos = mob->GetPosition();
        Vector vel = mob->GetVelocity();
        const double baseLat = 49.6117;
        const double baseLon = 6.1319;
        gps.lat      = baseLat + pos.x / 111000.0;
        gps.lon      = baseLon + pos.y / 91000.0;
        gps.altitude = static_cast<float>(pos.z);
        double spd   = std::sqrt(vel.x*vel.x + vel.y*vel.y);
        gps.speed_mps = static_cast<float>(spd);
        double hdg   = std::atan2(vel.y, vel.x) * 180.0 / M_PI;
        if (hdg < 0) hdg += 360.0;
        gps.heading_deg = static_cast<float>(hdg);
    }

    // 350 B 로 패딩. CAM 의 자원 점유를 ETSI 실측 평균 크기에 맞춘다.
    uint8_t buf[kCamPayloadBytes] = {0};
    const size_t copyLen = std::min(sizeof(GpsBroadcast),
                                    static_cast<size_t>(kCamPayloadBytes));
    std::memcpy(buf, &gps, copyLen);
    Ptr<Packet> pkt = Create<Packet>(buf, kCamPayloadBytes);
    if (sock->Send(pkt) > 0) { ++g_camTxOk; }
    else                     { ++g_camTxFail; }
    ++g_camAttempted;

    Simulator::Schedule(MilliSeconds(static_cast<uint64_t>(kCamIntervalMs)),
                        &SendBsmForNode, node, sock);
}

// GPS 브로드캐스트 수신 및 역직렬화
void NotifySlRxSizeDispatch(Ptr<Socket> socket)
{
    Ptr<Packet> pkt;
    Address from;
    while ((pkt = socket->RecvFrom(from))) {
        uint32_t sz = pkt->GetSize();
        if (sz > 1500 || sz < sizeof(GpsBroadcast)) continue;

        GpsBroadcast gps = {};
        pkt->CopyData(reinterpret_cast<uint8_t*>(&gps), sizeof(GpsBroadcast));

        g_rxCpPackets++;
        g_rxPacketsSl++;
        g_totalRxBitsSl += sz * 8;

        NS_LOG_DEBUG("[GPS-Rx] nodeId=" << gps.nodeId
                     << " lat=" << gps.lat << " lon=" << gps.lon
                     << " speed=" << gps.speed_mps << "m/s"
                     << " @ " << Simulator::Now().GetSeconds() << "s");
    }
}

void RxTraceCallback(Ptr<const Packet> packet)
{
    (void)packet;  // RSU PhyRxEnd는 SL 스루풋 집계에서 제외 (Uu 수신 오염 방지)
}

// ============================================================
// [G0] RTT ACK 지연 발송 — NR 수신 체인 밖에서 실행
// ============================================================
// NotifyTodRx 내부(NR RX callback chain)에서 직접 SendTo를 호출하면
// NR MAC UL 스케줄러가 "과거 슬롯"을 계산하여 Simulator::Schedule에
// 음수 delay가 전달되어 NS_ASSERT fail이 발생한다.
// Simulator::Schedule(MicroSeconds(1)) 로 현재 이벤트 체인 밖으로 꺼낸다.
void SendTodAck(uint32_t seq, Ipv4Address dst, uint8_t pathId)
{
    static uint32_t s_sendAckCallCount = 0;
    // ACK은 delay/loss 측정용 계측 신호이며 실제 데이터 전달 경로와 무관하게
    // 항상 Uu(제어 채널 성격)로 전송한다 — RSU가 PC5 역방향(carNode→remoteHost)
    // 릴레이를 지원하지 않는 구조적 제약(RSU의 SL 수신 콜백이 비활성화되어
    // 있고, RSU 라우팅 테이블에 8.0.0.0/8↔1.0.0.0/8을 잇는 경로가 없음) 및
    // 측정 목적상 어느 RAT으로 데이터가 수신됐는지와 무관하게 신뢰 가능한
    // 경로로만 보내면 충분하다는 판단에 따름. pathId(수신 RAT)는 더 이상
    // 소켓 선택에 사용하지 않는다.
    Ptr<Socket> ackSock = g_todAckSocketUu;
    ++s_sendAckCallCount;
    if (!ackSock) return;

    Ptr<Packet> ack = Create<Packet>(64);
    SeqTsHeader seqHdr;
    seqHdr.SetSeq(seq);
    ack->AddHeader(seqHdr);

    TvtMeasHeader meas;
    meas.type   = TVT_MEAS_ACK;
    meas.pathId = pathId;
    ack->AddHeader(meas);

    ackSock->SendTo(ack, 0, InetSocketAddress(dst, g_ackPort));
}

// 측정 프로브 응답 (carNode -> server, ACK 포트 공용).
// SendTodAck()과 같은 이유로 수신 RAT과 무관하게 항상 g_todAckSocketUu로
// 되돌린다 — RSU가 PC5 역방향 릴레이를 지원하지 않기 때문이며, 어느 경로를
// 측정했는지는 헤더의 pathId가 실어 나른다.
void SendMeasProbeResponse(uint8_t pathId, uint32_t probeSeq, Ipv4Address dst)
{
    Ptr<Socket> ackSock = g_todAckSocketUu;
    if (!ackSock) return;
    Ptr<Packet> resp = Create<Packet>(32);
    SeqTsHeader seqHdr;
    seqHdr.SetSeq(probeSeq);
    resp->AddHeader(seqHdr);
    TvtMeasHeader meas;
    meas.type   = TVT_MEAS_PROBE_RESP;
    meas.pathId = pathId;
    resp->AddHeader(meas);
    ackSock->SendTo(resp, 0, InetSocketAddress(dst, g_ackPort));
}

// ============================================================
// [G1] PC5 릴레이 ToD 수신 콜백: UE1 SL 포트 1235에서 RSU가 전달한 패킷 수신
// RSU가 MEC→RSU(Uu) ToD를 받아 225.1.0.0:1235 SL 멀티캐스트로 포워딩할 때
// UE1이 이 포트를 소켓으로 Listen하여 처리한다.
// ============================================================
// 수신 경로 공통 전달 함수 — Uu 수신과 PC5 수신 양쪽에서 호출한다.
static void DeliverTodPacket(Ptr<Packet> p, uint8_t rxPathId);

void NotifyUe1PcToD(Ptr<Socket> socket)
{
    Ptr<Packet> pkt;
    Address from;
    while ((pkt = socket->RecvFrom(from)))
    {
        Ptr<Packet> copy = pkt->Copy();

        // 측정 프로브는 데이터가 아니므로 먼저 걸러낸다. 프로브는 작고
        // TvtMeasHeader로 시작하므로 SeqTsSizeHeader 크기 검사보다 앞선다.
        TvtMeasHeader meas;
        if (copy->GetSize() >= meas.GetSerializedSize())
        {
            Ptr<Packet> peek = copy->Copy();
            peek->RemoveHeader(meas);
            if (meas.type == TVT_MEAS_PROBE)
            {
                SeqTsHeader probeHdr;
                if (peek->GetSize() >= probeHdr.GetSerializedSize())
                {
                    peek->PeekHeader(probeHdr);
                    Simulator::Schedule(MicroSeconds(1), &SendMeasProbeResponse,
                                        meas.pathId, probeHdr.GetSeq(),
                                        g_serverGlobalAddr);
                }
                continue;
            }
        }

        SeqTsSizeHeader seqTs;
        if (copy->GetSize() < seqTs.GetSerializedSize())
        {
            std::cout << "[CAR-PC5][WARN] pkt too small for SeqTsSizeHeader "
                      << "(size=" << copy->GetSize()
                      << "B, need=" << seqTs.GetSerializedSize() << "B) — 스킵 @ "
                      << Simulator::Now().GetSeconds() << "s" << std::endl;
            continue;
        }

        // PC5 경로 도착. 논리 패킷 중복 억제는 DeliverTodPacket() 한 곳에서만
        // 수행하므로 Uu 도착과 완전히 같은 규칙이 적용된다.
        DeliverTodPacket(copy, 1);
    }
}

// ============================================================
// [G] ToD 수신 콜백: E2E 지연 측정 (SeqTsSizeHeader 파싱)
// carNode의 PacketSink에 연결됨 — DL 방향 (server→carNode)
// ============================================================
// 한 physical copy가 도착했을 때의 공통 처리.
//
// 중복 전송된 두 copy 는 같은 logical application sequence number 를 갖는다.
// 따라서 여기서 seq 기준으로 정확히 한 번만 "logical delivery" 를 인정한다:
//
//   첫 유효 도착      -> logical delivery. rx 카운트/goodput/E2E delay/
//                        deadline 판정/ACK 이 여기서 딱 한 번 일어난다.
//   이후 같은 seq 도착 -> duplicate. logical 통계는 절대 건드리지 않고
//                        physical/resource 회계에만 반영한다.
//
// 경로별 측정(delay/throughput/RSSI)은 duplicate 에 대해서도 갱신한다 —
// 그것은 그 경로에서 실제로 관측된 값이기 때문이다.
static void
DeliverTodPacket(Ptr<Packet> p, uint8_t rxPathId)
{
    SeqTsSizeHeader seqTs;
    // 방어 코드: 손상/절단된 패킷이 PeekHeader 로 버퍼 밖을 읽지 않도록
    // 크기를 먼저 검증한다.
    if (!p || p->GetSize() < seqTs.GetSerializedSize()) {
        std::cout << "[ToD-Rx][WARN] pkt too small for SeqTsSizeHeader (size="
                  << (p ? p->GetSize() : 0) << "B, need=" << seqTs.GetSerializedSize()
                  << "B) — 스킵 @ " << Simulator::Now().GetSeconds() << "s" << std::endl;
        return;
    }
    p->PeekHeader(seqTs);

    const uint32_t seq         = seqTs.GetSeq();
    const double   txTime_s    = seqTs.GetTs().GetSeconds();
    const double   rxTime_s    = Simulator::Now().GetSeconds();
    const double   e2eDelay_ms = (rxTime_s - txTime_s) * 1000.0;
    const uint32_t slot        = (rxPathId == 1) ? 1 : 0;

    // ── 경로별 측정 (duplicate 여부와 무관하게 항상 갱신) ──────────────
    g_pathMeas[slot].NoteDelaySample(e2eDelay_ms, rxTime_s);
    g_pathLoss[slot].Note(true);

    double pktBits = p->GetSize() * 8.0;
    double todIntervalSec = (g_todPktSize + 20) * 8.0 / (g_todDataRateMbps * 1e6);
    if (slot == 1) {
        g_emaSlThpt = kAlpha * (pktBits / todIntervalSec) + (1.0 - kAlpha) * g_emaSlThpt;
        g_totalRxBitsSl += p->GetSize() * 8;
        g_rxPacketsSl++;
        g_slBytesWindow += p->GetSize();
    } else {
        g_emaUuThpt = kAlpha * (pktBits / todIntervalSec) + (1.0 - kAlpha) * g_emaUuThpt;
        g_totalRxBitsUu += p->GetSize() * 8;
        g_rxPacketsUu++;
        g_uuBytesWindow += p->GetSize();
    }
    g_pathMeas[slot].thptBps = (slot == 1) ? g_emaSlThpt : g_emaUuThpt;

    // ── duplicate 판정 ────────────────────────────────────────────────
    // Identity is checked against g_logicalSeen, which is never erased mid
    // episode. A copy that arrives after this seq's deadline already matured
    // is still a duplicate: it must not restore a delivery, a goodput byte,
    // an ACK or a deadline success.
    if (g_logicalSeen.find(seq) != g_logicalSeen.end())
    {
        ++g_duplicateCopiesRx;
        g_duplicateBytesRx += p->GetSize();
        if (g_tvtDeadlinePending.find(seq) == g_tvtDeadlinePending.end())
        {
            // deadline already matured for this seq -> late duplicate
            ++g_lateDuplicateRx;
            g_lateDuplicateBytes += p->GetSize();
        }
        return;
    }
    g_logicalSeen.insert(seq);

    // ── 여기부터는 logical packet 당 정확히 한 번만 실행된다 ──────────
    g_tvtLastRxTimeSec = rxTime_s;
    auto deadlineIt = g_tvtDeadlinePending.find(seq);
    if (deadlineIt != g_tvtDeadlinePending.end())
    {
        // T_delivery(k) = min(T_Uu(k), T_PC5(k)) — 먼저 도착한 유효 copy 가
        // 기준이다. 이후 다른 copy 가 늦게 오거나 아예 오지 않아도 이 logical
        // packet 에 별도의 deadline failure 를 추가하지 않는다.
        if (deadlineIt->second.deliveryDelayMs < 0.0)
        {
            deadlineIt->second.deliveryDelayMs = e2eDelay_ms;
        }
        if (e2eDelay_ms <= g_tvtDeadlineMs)
        {
            deadlineIt->second.deliveredInTime = true;
        }
    }

    g_packetDelayLog.push_back({seq, e2eDelay_ms, rxTime_s});

    g_recentE2eDelaysMs.push_back(e2eDelay_ms);
    g_delaySumMs += e2eDelay_ms;
    g_delayCount++;
    if (e2eDelay_ms > g_delayMaxMs) g_delayMaxMs = e2eDelay_ms;
    if (g_recentE2eDelaysMs.size() > kE2eDelayWindowSize)
    {
        g_recentE2eDelaysMs.pop_front();
    }

    // LEGACY_ONLY: D1/D2 누적. g_d1Sum/g_d2Sum 은 CSV 요약에만 쓰이고 env /
    // reward / action 으로 나가지 않는다.
    if (g_enableLegacyHoTelemetry)
    {
        if (g_interruptionActive) { g_d2Sum += e2eDelay_ms; g_d2Count++; }
        else                      { g_d1Sum += e2eDelay_ms; g_d1Count++; }
    }

    g_rxTodPackets++;
    ++g_logicalRxPackets;
    // useful goodput: 선언된 application payload byte 만, logical packet 당 한 번.
    // 이 줄은 duplicate 가 위에서 이미 return 한 뒤에만 도달한다.
    {
        const uint64_t payloadBytes = static_cast<uint64_t>(seqTs.GetSize());
        g_logicalGoodputBytesWindow += payloadBytes;
        g_logicalGoodputBytesTotal  += payloadBytes;
    }

    static uint32_t s_todRxCount = 0;
    if (++s_todRxCount % 50 == 1) {
        std::cout << "[ToD-Rx] carNode " << p->GetSize()
                  << "B E2E=" << std::fixed << std::setprecision(2) << e2eDelay_ms
                  << "ms path=" << (slot == 1 ? "PC5" : "Uu")
                  << " @ " << rxTime_s << "s  (logical#" << s_todRxCount << ")"
                  << std::endl;
    }

    // DL RX 중 TX 시도로 인한 PHY 크래시 방지.
    // ACK 은 항상 g_todAckSocketUu 로만 나가므로 그 존재 여부만 확인한다.
    // logical packet 당 한 번만 보낸다 — duplicate 는 위에서 이미 return 했다.
    if (g_todAckSocketUu && !g_disableTodAck) {
        Simulator::Schedule(MicroSeconds(1), &SendTodAck,
                            seq, g_serverGlobalAddr, rxPathId);
    }
}

void NotifyTodRx(Ptr<const Packet> packet, const Address& /*from*/, const Address& /*to*/)
{
    if (!packet) return;

    Ptr<Packet> copy = packet->Copy();

    // 측정 프로브 선별 — 데이터 경로와 같은 포트를 쓰므로 먼저 확인한다.
    TvtMeasHeader meas;
    if (copy->GetSize() >= meas.GetSerializedSize())
    {
        Ptr<Packet> peek = copy->Copy();
        peek->RemoveHeader(meas);
        if (meas.type == TVT_MEAS_PROBE)
        {
            SeqTsHeader probeHdr;
            if (peek->GetSize() >= probeHdr.GetSerializedSize())
            {
                peek->PeekHeader(probeHdr);
                Simulator::Schedule(MicroSeconds(1), &SendMeasProbeResponse,
                                    meas.pathId, probeHdr.GetSeq(),
                                    g_serverGlobalAddr);
            }
            return;
        }
    }

    // Uu 경로 도착. PC5 도착과 동일한 규칙으로 처리된다.
    DeliverTodPacket(copy, 0);
}

void
NotifyUuRlcPduRx(uint64_t /*imsi*/, uint16_t /*rnti*/, uint16_t /*txRnti*/,
                 uint8_t /*lcid*/, uint32_t size, double /*delay*/)
{
    g_rxPacketsUu++;
    g_totalRxBitsUu += size * 8;
}

// ============================================================
// [I] PHY 측정 콜백 (Uu RSRP / SINR)
// gNB attach 이후 활성화 → 핸드오버 판단에 사용
// ============================================================
// context 문자열에서 NodeList 번호 추출: "/NodeList/N/..."
static uint32_t
ExtractNodeId(const std::string& context)
{
    // format: /NodeList/N/DeviceList/...
    std::size_t start = context.find("/NodeList/");
    if (start == std::string::npos) return UINT32_MAX;
    start += 10; // skip "/NodeList/"
    std::size_t end = context.find('/', start);
    if (end == std::string::npos) return UINT32_MAX;
    return static_cast<uint32_t>(std::stoul(context.substr(start, end - start)));
}

void
SinrCallback(std::string context,
             uint16_t /*cellId*/, uint16_t /*rnti*/,
             double sinr, uint16_t bwpId)
{
    static uint32_t sinrCbCount = 0;
    if (sinrCbCount++ % 100 == 0)
        printf("[SINR] 콜백 호출 횟수=%u bwpId=%u\n", sinrCbCount, bwpId);

    if (bwpId > 1) return;  // BWP0(Uu), BWP1(협력인지 Uu) 모두 허용

    uint32_t nid = ExtractNodeId(context);
    if (nid != g_ue1NodeId) return;

    if (sinr != sinr) return;              // NaN 체크 (NaN != NaN)
    if (std::isinf(sinr)) return;          // inf 체크
    if (!std::isfinite(sinr) || sinr <= 0) return;

    // [수정] g_lastUuRssi는 더 이상 SINR 역산으로 구하지 않는다 — 아래
    // UuRssiCallback()이 NrInterference::RssiPerProcessedChunk 실측값으로
    // 직접 갱신한다. 이 콜백은 g_lastSinr(다른 state에서 사용) 수집만 유지.
    g_lastSinr     = 10.0 * std::log10(sinr);
    g_sinrMeasured = true;
    if (g_enablePc5Debug)
    {
        printf("[SINR-CB] nodeId=%u sinr=%.2f dB\n", nid, g_lastSinr);
    }
}

void
RsrpCallback(std::string context,
             uint16_t cellId, uint16_t imsi, uint16_t rnti,
             double rsrp, uint8_t bwpId)
{
    uint32_t nid = ExtractNodeId(context);
    if (nid != g_ue1NodeId) return;
    if (!std::isfinite(rsrp)) return;

    g_lastUuRsrp = rsrp;
}

// ============================================================
// [I-2] PC5 PSSCH 수신 콜백 — SL SINR 로깅 (진단용, RSSI 계산에는 더 이상 사용 안 함)
// RxPsschTraceUe 트레이스에 연결: SlRxDataPacketTraceParams.m_sinr (선형)
// ============================================================
void
SlPsschRxCallback(std::string context,
                  const SlRxDataPacketTraceParams params)
{
    // UE1(Car) 노드에서 수신된 PSSCH 만 사용
    if (g_ue1NodeId == UINT32_MAX) return;
    if (ExtractNodeId(context) != g_ue1NodeId) return;

    if (g_enablePc5Debug)
    {
        std::cout << "[SL-PHY] rx corrupt=" << params.m_corrupt
                  << " sinr=" << params.m_sinr
                  << " @ " << Simulator::Now().GetSeconds() << "s" << std::endl;
    }

    if (params.m_corrupt) return;
    if (params.m_sinr <= 0.0) return;

    // 선형 SINR → dB (진단/로그용으로만 유지, g_lastPc5Rssi 계산에는 미사용)
    g_slSinr_dB = 10.0 * std::log10(params.m_sinr);
}

// ============================================================
// [I-2b] Uu/PC5 RSSI 직접 측정 콜백
// contrib/nr/model의 NrInterference::RssiPerProcessedChunk (Uu) /
// NrSlInterference::SlRssiPerProcessedChunk (PC5, 신규 추가)에 직접 연결.
// 둘 다 noise+allSignals를 수신대역폭으로 적분한 실제 PHY 수신 전력(dBm)이며,
// SINR 역산·S-RSRP 대체·거리 기반 추정을 전혀 거치지 않는다.
// ============================================================
void
UuRssiCallback(double rssidBm)
{
    if (!std::isfinite(rssidBm)) return;
    g_lastUuRssi = rssidBm;

    static uint32_t s_count = 0;
    if (++s_count % 50 == 1)
    {
        std::cout << "[Uu-RSSI] rssi=" << std::fixed << std::setprecision(2)
                  << rssidBm << "dBm @ " << Simulator::Now().GetSeconds()
                  << "s (call#" << s_count << ")" << std::endl;
    }
}

void
Pc5RssiCallback(double rssidBm)
{
    if (g_enablePc5Debug)
    {
        std::cout << "[PC5-RSSI] t=" << Simulator::Now().GetSeconds()
                  << " rssi=" << rssidBm << std::endl;
    }

    if (!std::isfinite(rssidBm)) return;
    g_lastPc5Rssi = rssidBm;

    static uint32_t s_count = 0;
    if (++s_count % 50 == 1 && g_enablePc5Debug)
    {
        std::cout << "[PC5-RSSI] rssi=" << std::fixed << std::setprecision(2)
                  << rssidBm << "dBm @ " << Simulator::Now().GetSeconds()
                  << "s (call#" << s_count << ")" << std::endl;
    }
}

// ============================================================
// [RADIO-DIAG] PC5 SL MAC 스케줄러 계측 콜백 (원인 검증용, 로직 변경 없음)
// RSU의 NrSlUeMacScheduler에 직접 연결. 기존 GrantCreated/GrantPublished
// 트레이스와, 이번에 추가한 RlcBufferStatus 트레이스를 사용한다.
// ============================================================
void
Pc5GrantCreatedCallback(const NrSlUeMacScheduler::GrantInfo& grant, uint16_t /*psfchPeriod*/)
{
    if (grant.slotAllocations.empty()) return;
    uint32_t dst = grant.slotAllocations.begin()->dstL2Id;

    g_pc5GrantsThisSec++;
    if (g_pc5SeenDst.count(dst))
    {
        g_pc5ReselectionsThisSec++;
    }
    else
    {
        g_pc5SeenDst.insert(dst);
    }
    g_pc5LastNSelected = grant.nSelected;
    g_pc5LastRriMs     = grant.rri.GetMilliSeconds();
}

void
Pc5GrantPublishedCallback(const NrSlUeMac::NrSlGrant& grant, uint16_t /*psfchPeriod*/)
{
    g_pc5TbPublishedThisSec++;
    g_pc5TbBytesSumThisSec += grant.tbSize;
    g_pc5LastNSelected = grant.nSelected;
    g_pc5LastRriMs     = grant.rri.GetMilliSeconds();
}

void
Pc5RlcBufferStatusCallback(uint32_t /*dstL2Id*/, uint8_t /*lcid*/,
                           uint32_t txQueueSize, uint32_t retxQueueSize)
{
    g_pc5LastRlcQueueBytes  = txQueueSize;
    g_pc5LastRetxQueueBytes = retxQueueSize;
}

// ============================================================
// [RADIO-DIAG] PC5 링크버짓 계측 콜백 (SINR 저하 원인 분리용, 로직 변경 없음)
// carNode의 NrSlInterference::SlLinkBudget 트레이스에 연결.
// ============================================================
void
Pc5LinkBudgetCallback(double signalDbm, double interferenceDbm, double noiseDbm, double sinrDb,
                     uint32_t occupiedRbCount)
{
    g_pc5LastSignalDbm = signalDbm;
    g_pc5LastInterfDbm = interferenceDbm;
    g_pc5LastNoiseDbm  = noiseDbm;
    g_pc5LastSinrDb    = sinrDb;
    g_pc5SinrSumThisSec += sinrDb;
    g_pc5LinkBudgetSamplesThisSec++;

    // [RADIO-DIAG] A/B/C 실험 비교용 전체 런 누적(신호전력 유한값만 집계 — 간섭 0일 때의
    // -inf dBm interference는 signal/sinr 자체엔 영향 없으므로 그대로 포함)
    if (std::isfinite(signalDbm) && std::isfinite(sinrDb))
    {
        g_pc5RunSampleCount++;
        g_pc5RunSignalSum += signalDbm;
        g_pc5RunSignalMin  = std::min(g_pc5RunSignalMin, signalDbm);
        g_pc5RunSignalMax  = std::max(g_pc5RunSignalMax, signalDbm);
        g_pc5RunSinrSum   += sinrDb;
        g_pc5RunSinrMin    = std::min(g_pc5RunSinrMin, sinrDb);
        g_pc5RunSinrMax    = std::max(g_pc5RunSinrMax, sinrDb);

        // [RADIO-DIAG] PSCCH-like(좁은 대역)/PSSCH-like(넓은 대역) 분리 집계
        if (occupiedRbCount <= kNarrowRbThreshold)
        {
            g_narrowSampleCount++;
            g_narrowSignalSum += signalDbm;
            g_narrowSignalMin  = std::min(g_narrowSignalMin, signalDbm);
            g_narrowSignalMax  = std::max(g_narrowSignalMax, signalDbm);
        }
        else
        {
            g_wideSampleCount++;
            g_wideSignalSum += signalDbm;
            g_wideSignalMin  = std::min(g_wideSignalMin, signalDbm);
            g_wideSignalMax  = std::max(g_wideSignalMax, signalDbm);
        }
    }
}

// ============================================================
// [RADIO-DIAG] NLOSv 추가손실 / Shadowing 손실 계측 콜백
// ThreeGppV2vUrbanPropagationLossModel::NlosvAdditionalLoss,
// ThreeGppPropagationLossModel::ShadowingApplied 트레이스에 연결.
// ============================================================
void
Pc5NlosvLossCallback(double additionalLossDb)
{
    g_nlosvLossCount++;
    g_nlosvLossSum += additionalLossDb;
    g_nlosvLossMin  = std::min(g_nlosvLossMin, additionalLossDb);
    g_nlosvLossMax  = std::max(g_nlosvLossMax, additionalLossDb);
}

void
Pc5ShadowingCallback(double shadowingDb)
{
    g_shadowingCount++;
    g_shadowingSum += shadowingDb;
    g_shadowingMin  = std::min(g_shadowingMin, shadowingDb);
    g_shadowingMax  = std::max(g_shadowingMax, shadowingDb);
}

// ============================================================
// [RADIO-DIAG] ChannelConditionModel LOS/NLOS/NLOSv 판정 계측 콜백
// ThreeGppChannelConditionModel::LosConditionComputed 트레이스에 연결.
// (distance3D, pLos, pNlos, losCondition: 0=LOS, 1=NLOS, 2=NLOSv)
// ============================================================
void
Pc5LosConditionCallback(double /*distance3D*/, double /*pLos*/, double /*pNlos*/, int losCondition)
{
    if (losCondition == 0) g_losCount++;
    else if (losCondition == 1) g_nlosCount++;
    else if (losCondition == 2) g_nlosvCount++;
}

void
UuLosConditionCallback(double /*distance3D*/, double /*pLos*/, double /*pNlos*/, int losCondition)
{
    if (losCondition == 0) g_uuLosCount++;
    else if (losCondition == 1) g_uuNlosCount++;
    else if (losCondition == 2) g_uuNlosvCount++;

    static uint64_t s_uuLosLogCount = 0;
    if (++s_uuLosLogCount % 20 == 1) {
        const char* condStr = (losCondition == 0) ? "LOS" :
                              (losCondition == 1) ? "NLOS" : "NLOSv";
        std::cout << "[UU-LOS-DBG] t=" << Simulator::Now().GetSeconds()
                  << "s condition=" << condStr
                  << " (총 LOS=" << g_uuLosCount
                  << " NLOS=" << g_uuNlosCount
                  << " NLOSv=" << g_uuNlosvCount << ")" << std::endl;
    }
}

// ============================================================
// [RADIO-DIAG] PC5 링크버짓 1초 주기 출력
// 차량-RSU 거리, TxPower(고정 상수), Noise Power(고정 계산값), 그리고
// SlLinkBudget 트레이스로 실측한 신호/간섭/잡음/SINR을 함께 출력한다.
// ============================================================
void
PrintPc5LinkBudgetStats()
{
    double now = Simulator::Now().GetSeconds();

    double distM = -1.0;
    if (g_carNodePtr && g_rsuNodesGlobal.GetN() > 0)
    {
        uint32_t nearIdx = FindNearestRsu(g_rsuNodesGlobal, g_carNodePtr);
        Ptr<MobilityModel> rsuMob = g_rsuNodesGlobal.Get(nearIdx)->GetObject<MobilityModel>();
        Ptr<MobilityModel> carMob = g_carNodePtr->GetObject<MobilityModel>();
        if (rsuMob && carMob)
        {
            distM = CalculateDistance(rsuMob->GetPosition(), carMob->GetPosition());
        }
    }

    double avgSinrDb = (g_pc5LinkBudgetSamplesThisSec > 0)
        ? g_pc5SinrSumThisSec / g_pc5LinkBudgetSamplesThisSec
        : NAN;

    // Noise Power(dBm) = 10*log10(kT0 * 10^(NF/10) * B * 1000), B = SL BWP 대역폭(Hz).
    // NoiseFigure=7.0dB(main()에서 SetGnbPhyAttribute/SetUePhyAttribute로 설정),
    // kT0 = -174 dBm/Hz. 참고용 상수이며 매 스텝 재계산할 필요 없는 값이다.
    static const double kNoiseFigureDb = 7.0;
    static const double kThermalNoiseDbmHz = -174.0;

    std::cout << "[PC5-LINKBUDGET @ " << std::fixed << std::setprecision(1) << now << "s] "
              << "dist_rsu_car=" << std::setprecision(2) << distM << "m"
              << "  tx_power_rsu=" << PC5_RSU_TXPOWER_DBM << "dBm"
              << "  tx_power_ue=" << PC5_UE_TXPOWER_DBM << "dBm"
              << "  ant_gain_tx=0dBi(quasi-omni,고정)"
              << "  ant_gain_rx=0dBi(quasi-omni,고정)"
              << "  noise_figure=" << kNoiseFigureDb << "dB"
              << "  thermal_floor=" << kThermalNoiseDbmHz << "dBm/Hz"
              << "  signal(last)=" << std::setprecision(2) << g_pc5LastSignalDbm << "dBm"
              << "  interference(last)=" << g_pc5LastInterfDbm << "dBm"
              << "  noise(last)=" << g_pc5LastNoiseDbm << "dBm"
              << "  sinr(last)=" << g_pc5LastSinrDb << "dB"
              << "  sinr(avg/s)=" << avgSinrDb << "dB"
              << "  samples/s=" << g_pc5LinkBudgetSamplesThisSec
              << "  effective_loss=" << (PC5_RSU_TXPOWER_DBM - g_pc5LastSignalDbm) << "dB"
              << std::endl;

    g_pc5SinrSumThisSec = 0.0;
    g_pc5LinkBudgetSamplesThisSec = 0;

    if (now < Simulator::GetMaximumSimulationTime().GetSeconds() - 0.5)
        Simulator::Schedule(Seconds(1.0), &PrintPc5LinkBudgetStats);
}

// ============================================================
// [RADIO-DIAG] PC5 SL MAC 스케줄러 계측치 1초 주기 출력
// ============================================================
void
PrintPc5SchedulerStats()
{
    double now = Simulator::Now().GetSeconds();

    double avgTbBytes = (g_pc5TbPublishedThisSec > 0)
        ? static_cast<double>(g_pc5TbBytesSumThisSec) / g_pc5TbPublishedThisSec
        : 0.0;
    // 이론적 최대 처리량(Mbps) = 초당 발행된 TB 수 × TB당 평균 바이트 × 8 / 1e6
    // (재전송 예약 슬롯은 새 데이터를 싣지 않으므로 tbSize에는 포함되지 않음)
    double theoreticalMbps = (g_pc5TbPublishedThisSec * avgTbBytes * 8.0) / 1e6;

    std::cout << "[PC5-SCHED @ " << std::fixed << std::setprecision(1) << now << "s] "
              << "grants/s=" << g_pc5GrantsThisSec
              << "  resel/s=" << g_pc5ReselectionsThisSec
              << "  tb_published/s=" << g_pc5TbPublishedThisSec
              << "  avg_tb_bytes=" << std::setprecision(1) << avgTbBytes
              << "  nSelected(last)=" << static_cast<int>(g_pc5LastNSelected)
              << "  rri_ms(last)=" << std::setprecision(1) << g_pc5LastRriMs
              << "  rlc_tx_queue=" << g_pc5LastRlcQueueBytes << "B"
              << "  rlc_retx_queue=" << g_pc5LastRetxQueueBytes << "B"
              << "  theoretical_max=" << std::setprecision(3) << theoreticalMbps << "Mbps"
              << std::endl;

    g_pc5GrantsThisSec       = 0;
    g_pc5ReselectionsThisSec = 0;
    g_pc5TbPublishedThisSec  = 0;
    g_pc5TbBytesSumThisSec   = 0;

    if (now < Simulator::GetMaximumSimulationTime().GetSeconds() - 0.5)
        Simulator::Schedule(Seconds(1.0), &PrintPc5SchedulerStats);
}

// ============================================================
// [I-4b] ns3-ai 공유 메모리 초기화 유틸리티
// ============================================================

/** Python tvt_test.py 가 먼저 공유 메모리를 생성한 뒤 이 함수로 접속한다.
 *  실패 시 g_aiInterface = nullptr → CollectV2xState 에서 action=0(Uu).
 *  Modified: Action semantics changed (0=Uu, 1=PC5) */
static void
ConnectAiInterface()
{
    try {
        auto* iface = Ns3AiMsgInterface::Get();
        iface->SetIsMemoryCreator(false);   // Python 이 creator
        iface->SetUseVector(false);
        iface->SetHandleFinish(false);      // done 필드로 수동 관리
        iface->SetNames(g_shmSegment, "TvtAiEnv", "TvtAiAct", "TvtAiSync");
        g_aiInterface = iface->GetInterface<TvtAiEnv, TvtAiAct>();
        std::cout << "[ns3-ai] 공유 메모리 연결 성공 (seg=" << g_shmSegment
                  << ")" << std::endl;
    } catch (const std::exception& e) {
        // Modified: Action semantics changed (0=Uu, 1=PC5)
        std::cerr << "[ns3-ai] 공유 메모리 연결 실패: " << e.what()
                  << " — Python DDQN 없이 action=0(Uu) 유지" << std::endl;
        g_aiInterface = nullptr;
    }
}

/** 시뮬레이션 종료 신호(done=1)를 Python 에 전달한다.
 *  Python 은 이 메시지를 받으면 루프를 탈출하고 모델을 저장한다. */
static void
SendAiDone()
{
    if (!g_aiInterface) return;
    g_aiInterface->CppSendBegin();
    auto* env = g_aiInterface->GetCpp2PyStruct();
    if (g_hasLastSentAiEnv)
    {
        *env = g_lastSentAiEnv;
    }
    else
    {
        *env = TvtAiEnv{};
        env->contract_version = TVT_AI_CONTRACT_VERSION;
        env->simulation_time_s = Simulator::Now().GetSeconds();
        env->uu_rssi = g_pathMeas[0].rssiDbm;
        env->pc5_rssi = g_pathMeas[1].rssiDbm;
        env->current_action = g_currentAction;
        // Keep the fallback env well formed rather than zero-filled. This
        // message is a handshake boundary and never enters reward or replay.
        FillMobilityContext(env);
    }
    // 종료 메시지는 환경 전이가 아니라 semaphore handshake용 경계다.
    // Python은 ACK를 돌려주되 reward/replay에는 절대 넣지 않는다.
    env->decision_id = 0;
    env->applied_decision_id = 0;
    env->transition_valid = 0;
    // The done message replays the last env, so the cohort it carried must be
    // cleared: republishing it would credit the same cohort twice.
    env->cohort_ready = 0;
    env->cohort_backlog = static_cast<uint32_t>(g_tickCohorts.size());
    env->done = 1;
    g_aiInterface->CppSendEnd();
    // Python 이 done=1 을 수신·처리하고 더미 액션을 보낼 때까지 대기
    // (이렇게 해야 Python 이 PyRecvEnd 를 완료한 뒤 NS-3 가 종료되어
    //  다음 에피소드 NS-3 시작 시 공유 메모리 상태가 깨끗하게 유지됨)
    g_aiInterface->CppRecvBegin();
    (void)g_aiInterface->GetPy2CppStruct()->action;  // 더미 액션 읽기 (무시)
    g_aiInterface->CppRecvEnd();
    g_aiInterface = nullptr;
    std::cout << "[ns3-ai] 종료 신호 전송 완료" << std::endl;
}

// ============================================================
// [I-4b] MOBILITY CONTEXT  c_t  (observation contract 8)
//
// Road-plane kinematics of the ego vehicle, published for the UPPER tier
// (Temporal Transformer reward designer + GoalCritic). These are NOT part of
// the lower DDQN radio state: that stays at 7 dims and never sees mobility.
//
// Both quantities are STRICT_EXOGENOUS. They are mobility-model state driven
// by the replayed SUMO trace, so the duplication action cannot move them, and
// they are read before this tick's goal and action are chosen.
//
// No missing-value encoding: once mobility is installed both are always
// defined, so an impossible missing condition aborts loudly rather than being
// folded into a valid numeric extreme (0 m/s and 0 m are both VALID values).
// ============================================================
static void
FillMobilityContext(TvtAiEnv* env)
{
    NS_ABORT_MSG_IF(!g_carNodePtr, "mobility context: ego node is null");
    Ptr<MobilityModel> carMob = g_carNodePtr->GetObject<MobilityModel>();
    NS_ABORT_MSG_IF(!carMob, "mobility context: ego has no MobilityModel");
    NS_ABORT_MSG_IF(g_rsuNodesGlobal.GetN() == 0,
                    "mobility context: no RSU installed");
    Ptr<MobilityModel> rsuMob =
        g_rsuNodesGlobal.Get(0)->GetObject<MobilityModel>();
    NS_ABORT_MSG_IF(!rsuMob, "mobility context: RSU has no MobilityModel");

    const Vector pos = carMob->GetPosition();
    const Vector vel = carMob->GetVelocity();
    // GetVelocity() is exact for the trace-driven ConstantVelocityMobilityModel
    // installed by Ns2MobilityHelper -- never a finite-difference estimate.
    env->ego_speed_mps = std::sqrt(vel.x * vel.x + vel.y * vel.y);

    // 2-D horizontal distance to THE RSU. numRsu == 1, so this is not a
    // nearest-of-many selection. RSU height is excluded deliberately: the
    // semantic is road-plane geometry, not link geometry.
    const Vector r  = rsuMob->GetPosition();
    const double dx = pos.x - r.x;
    const double dy = pos.y - r.y;
    env->ego_rsu_distance_m = std::sqrt(dx * dx + dy * dy);
}

// ============================================================
// [I-5] V2X 상태 수집 루프 (10ms 주기)
//
// 수집 항목:
//   uu_rssi  — g_lastUuRssi (RsrpCallback 갱신)
//   pc5_rssi — g_lastPc5Rssi (SlPsschRxCallback 갱신)
//   currentAction — g_currentAction 동기화
//
// NOTE (contract 8): ego speed is NOT collected here and is NOT a lower-tier
// state variable. It is published separately as part of the UPPER-tier
// mobility context c_t by FillMobilityContext(); see [I-4b]. The retired 4-D
// state that once listed speed here no longer exists.
//
// 보상 계산용:
//   uuThroughput_kbps / pc5Throughput_kbps — EMA(α=0.3) 10ms 윈도우
// ============================================================
static void
CollectV2xState()
{

    double now = Simulator::Now().GetSeconds();

    for (auto it = g_tvtDeadlinePending.begin(); it != g_tvtDeadlinePending.end();)
    {
        if ((now - it->second.txTimeSec) * 1000.0 < g_tvtDeadlineMs)
        {
            ++it;
            continue;
        }
        // 이 map 은 logical seq 로 키가 잡혀 있으므로, 중복으로 두 copy 를
        // 보냈더라도 logical packet 하나당 정확히 한 번만 성숙한다. 따라서
        // 하나의 logical packet 이 success 와 failure 로 동시에 잡히는 일이
        // 구조적으로 불가능하다.
        g_tvtDeadlineOutcomeWindow.push_back(it->second.deliveredInTime);
        if (it->second.deliveredInTime) ++g_logicalDeadlineSuccess;
        else                            ++g_logicalDeadlineMiss;
        // ── cohort attribution ─────────────────────────────────────────
        // Credit the outcome to the decision tick that TRANSMITTED this
        // packet, never to the tick its deadline happens to ripen in.
        if (it->second.hasCohort)
        {
            auto cohortIt = g_tickCohorts.find(it->second.cohortSerial);
            if (cohortIt != g_tickCohorts.end())
            {
                TvtTickCohort& cohort = cohortIt->second;
                if (cohort.pendingDeadlines > 0) --cohort.pendingDeadlines;
                if (it->second.deliveredInTime) ++cohort.success;
                else                            ++cohort.miss;
                if (it->second.deliveryDelayMs >= 0.0)
                {
                    cohort.delaySumMs += it->second.deliveryDelayMs;
                    ++cohort.delayCount;
                }
            }
        }
        if (g_tvtDeadlineOutcomeWindow.size() > g_tvtDeadlineWindowSize)
        {
            g_tvtDeadlineOutcomeWindow.pop_front();
        }
        // Only the DEADLINE lifetime ends here. The logical identity stays in
        // g_logicalSeen so a late-arriving copy is still recognised as a
        // duplicate (this is the A-2 fix: erasing identity here used to let a
        // late copy be counted as a fresh logical delivery).
        it = g_tvtDeadlinePending.erase(it);
    }

    double sampleDurationSec = now - g_tvtLastKpiSampleTimeSec;
    if (sampleDurationSec > 0.0)
    {
        g_tvtUuThroughputBps =
            static_cast<double>(g_uuBytesWindow) * 8.0 / sampleDurationSec;
        g_tvtPc5ThroughputBps =
            static_cast<double>(g_slBytesWindow) * 8.0 / sampleDurationSec;
        // 동일 window, 동일 reset 지점 — lower reward 가 읽는 dense 측정이
        // 경로 관측과 다른 구간을 보지 않도록 한다.
        g_tvtLogicalGoodputBps =
            static_cast<double>(g_logicalGoodputBytesWindow) * 8.0
            / sampleDurationSec;
    }
    g_uuBytesWindow = 0;
    g_slBytesWindow = 0;
    g_logicalGoodputBytesWindow = 0;
    g_tvtLastKpiSampleTimeSec = now;

    // ── Uu/PC5 측정값 (RSSI 콜백에서 갱신된 전역 변수 읽기) ───────
    // 거리 기반 RSSI 추정 fallback 제거 — g_lastPc5Rssi는 Pc5RssiCallback()
    // (NrSlInterference::SlRssiPerProcessedChunk 실측)만이 갱신한다. 아직
    // 수신 이벤트가 없는 초기 구간에는 임의 대체값 없이 초기값이 유지된다.
    g_dqnState.uu_rssi = g_lastUuRssi;
    g_dqnState.pc5_rssi_avg = g_lastPc5Rssi;
    g_pathMeas[0].rssiDbm = g_lastUuRssi;
    g_pathMeas[1].rssiDbm = g_lastPc5Rssi;

    // ── Reward telemetry 계산 ──────────────────────────────
    double e2eDelayAvgMs = -1.0;
    if (!g_recentE2eDelaysMs.empty())
    {
        double sum = 0.0;
        for (double d : g_recentE2eDelaysMs) sum += d;
        e2eDelayAvgMs = sum / static_cast<double>(g_recentE2eDelaysMs.size());
    }

    // ── LEGACY_ONLY telemetry ──────────────────────────────────────────
    // D1/D2 delay bucketing and the Scenario1/2/3 CSVs below are the MP-QUIC-
    // era "handover" instrumentation. They are kept so those historical runs
    // stay reproducible and their CSVs stay readable, and they are NOT part
    // of the dual-connectivity contract: none of these values is sent to
    // Python, enters any objective, or gates any delivery. The new campaign
    // reports switching neutrally as switch_count / time_on_action, and makes
    // no "handover latency" claim -- the measured D2-D1 difference is 0.
    // 감시구간 종료 처리 + D2-D1 차이 계산 (LEGACY_ONLY, 기본 OFF)
    if (g_enableLegacyHoTelemetry &&
        g_interruptionActive &&
        (now - g_interruptionStartSimTime) >= kInterruptionWindowSec)
    {
        double d1Avg = (g_d1Count > 0) ? g_d1Sum / static_cast<double>(g_d1Count) : 0.0;
        double d2Avg = (g_d2Count > 0) ? g_d2Sum / static_cast<double>(g_d2Count) : 0.0;
        double hoDelayMs = std::max(0.0, d2Avg - d1Avg);
        g_interruptionActive = false;
        g_hoEventIndex++;
        g_hoDelayLog.push_back({g_hoEventIndex, hoDelayMs});
    }

    // [ACK 기반 delay/loss 측정 — DDQN 전달용] 최근 g_ackOutcomeWindow(최대
    // kAckLossWindowSize개) 중 타임아웃(false) 비율. 표본이 없으면 0.0.
    double ackLossRate = 0.0;
    if (!g_ackOutcomeWindow.empty())
    {
        uint32_t lostCount = 0;
        for (bool ok : g_ackOutcomeWindow) if (!ok) ++lostCount;
        ackLossRate = static_cast<double>(lostCount) / static_cast<double>(g_ackOutcomeWindow.size());
    }

    // 같은 100ms의 tx/rx 차이는 in-flight 패킷을 손실로 오인한다. 완료된
    // ACK/timeout cohort만 손실 표본으로 쓰며, 표본 유효성을 별도로 전달한다.
    const bool delayValid = e2eDelayAvgMs >= 0.0 && std::isfinite(e2eDelayAvgMs);
    const bool deadlineLossValid = !g_tvtDeadlineOutcomeWindow.empty();
    uint32_t deadlineMisses = 0;
    for (bool delivered : g_tvtDeadlineOutcomeWindow)
    {
        if (!delivered)
        {
            ++deadlineMisses;
        }
    }
    const double deadlineLossRate =
        deadlineLossValid
            ? static_cast<double>(deadlineMisses) / g_tvtDeadlineOutcomeWindow.size()
            : 0.0;
    const double timeSinceLastRxMs =
        (g_tvtLastRxTimeSec >= 0.0) ? (now - g_tvtLastRxTimeSec) * 1000.0 : 1e9;
    double timeSinceLastSwitchMs =
        (g_lastActionSwitchTimeSec >= 0.0)
            ? (now - g_lastActionSwitchTimeSec) * 1000.0
            : 1e9;
    // No hold is computed: there is nothing to hold back. Time since the last
    // switch is reported as a diagnostic only.

    // ── close the open tick cohort ─────────────────────────────────────
    // The throughput / outage window that just ended is exactly the window
    // this cohort's packets were transmitted in, so the snapshot is taken
    // here, before the next decision opens a new cohort.
    if (g_haveOpenCohort)
    {
        auto openIt = g_tickCohorts.find(g_openCohortSerial);
        if (openIt != g_tickCohorts.end())
        {
            openIt->second.closed = true;
            openIt->second.closeTimeSec = now;
            openIt->second.uuThptBps = g_tvtUuThroughputBps;
            openIt->second.pc5ThptBps = g_tvtPc5ThroughputBps;
            openIt->second.logicalGoodputBps = g_tvtLogicalGoodputBps;
            openIt->second.timeSinceLastRxMs = timeSinceLastRxMs;
        }
        g_haveOpenCohort = false;
    }

    // ── publish the oldest matured cohort ──────────────────────────────
    // Cohorts open one per decision and their packets mature in order, so at
    // most one becomes publishable per decision and FIFO order is exact.
    TvtTickCohort maturedCohort;
    bool haveMaturedCohort = false;
    if (!g_tickCohorts.empty())
    {
        auto oldest = g_tickCohorts.begin();
        if (oldest->second.Complete())
        {
            maturedCohort = oldest->second;
            haveMaturedCohort = true;
            g_tickCohorts.erase(oldest);
            ++g_cohortsPublished;
        }
    }
    // Guard only: a cohort can never stall, because every deadline matures
    // within g_tvtDeadlineMs of transmission. Bounded so a bug cannot leak.
    while (g_tickCohorts.size() > kMaxLiveCohorts)
    {
        g_tickCohorts.erase(g_tickCohorts.begin());
        ++g_cohortsDropped;
    }

    // ── 현재 적용 중인 action 동기화 ────────────────────────
    g_dqnState.currentAction = g_currentAction;

    // ── action 결정: ratMode 에 따라 목표 duplication level 결정 ──
    // ddqn: Python(ns3-ai) 응답 사용. dup0..dup100: Python 없이 고정 level.
    // Python 미연결 → requestedAction = DUP_0 (중복 없음 유지)
    int    requestedAction = TVT_AI_ACTION_DUP_0;
    int    appliedAction = g_currentAction;
    uint64_t currentDecisionId = 0;
    bool   actionResponseValid = true;
    bool   haveMeasAnchors = false;
    double s2_D1 = 0.0, s3_sendWall = 0.0, s1_inferWall = 0.0, s3_actionWall = 0.0;
    uint64_t receivedMacroId = TVT_AI_NO_MACRO;
    int32_t  receivedMacroTick = -1;

    // ratMode 는 Python 없이 duplication level 을 강제하는 테스트 훅이다.
    // Forced Duplication Test(F0..F100)가 이 경로를 쓴다.
    if (g_forcedDuplicationLevel >= 0)
    {
        requestedAction = g_forcedDuplicationLevel;
    }
    else if (g_aiInterface)
    {
        // [S2] CppSendBegin 직전 시뮬 시각 (D1)
        s2_D1       = now;
        // [S3] CppSendBegin 직전 wall-clock — S3 측정 시작점
        s3_sendWall = ScenarioWallNow();

        // 직전 decision 이후 실제 전환이 있었는지(비동기 CheckAndSwitchInterface가
        // 적용한 결과) g_switchCount 스냅샷 비교로 판단 → is_switch로 전송.
        int isSwitchSinceLastDecision =
            (g_switchCount != g_switchCountAtLastDecision) ? 1 : 0;
        g_switchCountAtLastDecision = g_switchCount;

        currentDecisionId = g_nextAiDecisionId++;
        g_aiInterface->CppSendBegin();
        auto* env          = g_aiInterface->GetCpp2PyStruct();
        env->contract_version = TVT_AI_CONTRACT_VERSION;
        env->simulation_time_s = now;

        // ── 경로별 측정: 두 경로 모두 매 결정마다 전달한다 ────────────
        // 현재 action 이 어느 쪽이든 상관없이 갱신된다. 데이터가 흐르는
        // 경로는 실제 도착으로, 흐르지 않는 경로는 100ms 측정 프로브로
        // 유지된다. 값이 한 번도 측정되지 않았다면 valid=0 이며, Python 은
        // 그 사실을 그대로 볼 수 있다.
        env->uu_rssi     = g_pathMeas[0].rssiDbm;
        env->pc5_rssi    = g_pathMeas[1].rssiDbm;
        env->uu_delay_ms  = g_pathMeas[0].delayMs;
        env->pc5_delay_ms = g_pathMeas[1].delayMs;
        env->uu_loss_rate  = g_pathLoss[0].Rate();
        env->pc5_loss_rate = g_pathLoss[1].Rate();
        env->uu_throughput_bps = g_tvtUuThroughputBps;
        env->pc5_throughput_bps = g_tvtPc5ThroughputBps;
        env->logical_goodput_bps = g_tvtLogicalGoodputBps;
        env->uu_measurement_age_ms  = g_pathMeas[0].AgeMs(now);
        env->pc5_measurement_age_ms = g_pathMeas[1].AgeMs(now);

        // ── mobility context c_t (contract 8) ─────────────────────────────
        // Sampled here, i.e. strictly BEFORE Python selects this tick's goal
        // and action from this very env.
        FillMobilityContext(env);
        env->uu_measurement_valid  = g_pathMeas[0].valid ? 1 : 0;
        env->pc5_measurement_valid = g_pathMeas[1].valid ? 1 : 0;

        env->current_action = g_dqnState.currentAction;
        env->action_hold_enabled = kActionHoldEnabled ? 1 : 0;
        env->decision_id = currentDecisionId;
        env->applied_decision_id = g_lastAiResult.decisionId;
        env->requested_action = g_lastAiResult.requested;
        env->applied_action = g_lastAiResult.applied;
        env->apply_status = static_cast<int32_t>(g_lastAiResult.status);
        env->transition_valid = g_lastAiResult.valid ? 1 : 0;

        env->e2e_delay_ms = e2eDelayAvgMs;
        env->deadline_loss_rate = deadlineLossRate;
        env->delay_valid = delayValid ? 1 : 0;
        env->deadline_loss_valid = deadlineLossValid ? 1 : 0;
        env->time_since_last_rx_ms = timeSinceLastRxMs;

        // ── 논리 패킷 회계 (중복 억제 후) ─────────────────────────────
        env->logical_tx_packets = g_logicalTxPackets;
        env->logical_rx_packets = g_logicalRxPackets;
        env->logical_deadline_success = g_logicalDeadlineSuccess;
        env->logical_deadline_miss = g_logicalDeadlineMiss;

        // ── 물리/중복 회계 (resource cost 근거) ───────────────────────
        // window_* 는 직전 결정 구간에서 실제로 소켓에 넘어간 copy 수다.
        // 선택한 action 이 아니라 이 실측값으로 cost 를 계산해야 한다.
        env->attempted_physical_copies_tx = g_attemptedCopiesTx;
        env->successful_physical_copies_tx = g_successfulCopiesTx;
        env->duplicate_copies_rx = g_duplicateCopiesRx;
        env->duplicate_bytes_rx = g_duplicateBytesRx;
        env->duplicated_logical_packets = g_duplicatedLogicalPackets;
        env->window_logical_tx = g_windowLogicalTx;
        env->window_attempted_copies = g_windowAttemptedCopies;
        env->window_successful_copies = g_windowSuccessfulCopies;
        env->window_duplicated_logical_packets =
            g_windowDuplicatedLogicalPackets;

        // ── matured tick cohort (macro-attributed reliability) ────────
        env->cohort_ready = haveMaturedCohort ? 1 : 0;
        env->cohort_macro_id = maturedCohort.macroId;
        env->cohort_macro_tick = maturedCohort.macroTick;
        env->cohort_action = maturedCohort.action;
        env->cohort_logical_tx = maturedCohort.logicalTx;
        env->cohort_deadline_success = maturedCohort.success;
        env->cohort_deadline_miss = maturedCohort.miss;
        env->cohort_delay_sum_ms = maturedCohort.delaySumMs;
        env->cohort_delay_count = maturedCohort.delayCount;
        env->cohort_attempted_copies = maturedCohort.attemptedCopies;
        env->cohort_duplicated_logical_packets =
            maturedCohort.duplicatedLogicalPackets;
        env->cohort_uu_throughput_bps = maturedCohort.uuThptBps;
        env->cohort_pc5_throughput_bps = maturedCohort.pc5ThptBps;
        env->cohort_logical_goodput_bps = maturedCohort.logicalGoodputBps;
        env->cohort_time_since_last_rx_ms = maturedCohort.timeSinceLastRxMs;
        env->cohort_backlog = static_cast<uint32_t>(g_tickCohorts.size());

        // 0 during observation warm-up: Python must not store transitions or
        // count KPI from these steps, but it DOES fill the context history.
        env->formal_accounting_active =
            (now >= g_trafficStartSec) ? 1 : 0;
        env->time_since_last_switch_ms  = timeSinceLastSwitchMs;
        env->is_switch                  = isSwitchSinceLastDecision;
        env->done          = 0;  // done=1 은 SendAiDone() 에서만 전송
        g_lastSentAiEnv = *env;
        g_hasLastSentAiEnv = true;
        g_aiInterface->CppSendEnd();
        // 다음 결정 구간을 위해 window 누적치를 비운다.
        g_windowLogicalTx = 0;
        g_windowAttemptedCopies = 0;
        g_windowSuccessfulCopies = 0;
        g_windowDuplicatedLogicalPackets = 0;

        g_aiInterface->CppRecvBegin();
        auto* act = g_aiInterface->GetPy2CppStruct();
        int receivedAction = static_cast<int>(act->action);
        uint64_t receivedDecisionId = act->decision_id;
        uint32_t receivedContractVersion = act->contract_version;
        // Goal macro this decision belongs to. ns-3 does not interpret it; it
        // only stamps the cohort so the deadline outcome can be credited back.
        receivedMacroId = act->macro_id;
        receivedMacroTick = act->macro_tick;
        s1_inferWall  = act->infer_start_wall_s; // [S1/S3]
        s3_actionWall = act->action_sent_wall_s; // [S3]
        g_aiInterface->CppRecvEnd();
        haveMeasAnchors = true;

        if (receivedContractVersion != TVT_AI_CONTRACT_VERSION)
        {
            actionResponseValid = false;
            requestedAction = g_dqnState.currentAction;
            g_lastAiResult = {currentDecisionId, receivedAction, g_dqnState.currentAction,
                              TVT_AI_APPLY_CONTRACT_MISMATCH, false};
        }
        else if (receivedDecisionId != currentDecisionId)
        {
            actionResponseValid = false;
            requestedAction = g_dqnState.currentAction;
            g_lastAiResult = {currentDecisionId, receivedAction, g_dqnState.currentAction,
                              TVT_AI_APPLY_DECISION_ID_MISMATCH, false};
        }
        else if (receivedAction >= 0 && receivedAction < TVT_AI_ACTION_COUNT)
        {
            requestedAction = receivedAction;
        }
        else
        {
            actionResponseValid = false;
            requestedAction = g_dqnState.currentAction;
            g_invalidActionCount++;
            g_lastAiResult = {currentDecisionId, receivedAction, g_dqnState.currentAction,
                              TVT_AI_APPLY_INVALID_ACTION, false};
        }
    }

    // ── ACTION 적용: hold 없음 ─────────────────────────────────────────
    // Every well-formed decision is applied verbatim. The only ways requested
    // and applied can differ are a malformed response (invalid action,
    // decision-id mismatch, contract mismatch), which is handled above by
    // falling back to the action already in force and marking the transition
    // invalid. There is no policy-level veto of any kind.
    appliedAction = requestedAction;
    if (g_ratMode == "ddqn" && g_aiInterface && actionResponseValid)
    {
        if (requestedAction == g_dqnState.currentAction)
        {
            g_lastAiResult = {currentDecisionId, requestedAction, appliedAction,
                              TVT_AI_APPLY_STAY, true};
        }
        else
        {
            g_lastAiResult = {currentDecisionId, requestedAction, g_dqnState.currentAction,
                              TVT_AI_APPLY_PENDING, false};
            g_pendingAiDecision = true;
            g_pendingAiDecisionId = currentDecisionId;
            g_pendingAiRequestedAction = requestedAction;
        }
    }

    // [진단용] requested != applied 는 이제 프로토콜 오류(malformed response)
    // 에서만 발생할 수 있다. 정상 동작에서는 0 이어야 한다.
    static uint32_t s_actionMismatchCount = 0;
    if (requestedAction != appliedAction)
    {
        ++s_actionMismatchCount;
        std::cout << "[ACTION-MISMATCH][PROTOCOL-ERROR] requested=" << requestedAction
                  << " applied=" << appliedAction
                  << " current_action=" << g_dqnState.currentAction
                  << " status=" << AiApplyStatusName(g_lastAiResult.status)
                  << " count=" << s_actionMismatchCount << std::endl;
    }

    // targetInterface(appliedAction) != currentInterface 인 경우에만 실제 전환이
    // 일어날 수 있으므로, 이 경우에만(ddqn 모드+Python 응답 수신 시) 측정값을 저장한다.
    if (haveMeasAnchors && appliedAction != g_dqnState.currentAction)
    {
        g_pendingHoMeas       = true;
        g_pend_s1_inferWall   = s1_inferWall;
        g_pend_s2_D1          = s2_D1;
        g_pend_s3_sendWall    = s3_sendWall;
        g_pend_s3_inferWall   = s1_inferWall;
        g_pend_s3_actionWall  = s3_actionWall;
    }

    if (requestedAction >= 0 && requestedAction < TVT_AI_ACTION_COUNT)
        g_levelSelectedCount[requestedAction]++;

    // ── DDQN 액션 → PM 반영 ─────────────────────────────────
    g_pathTx.SetAction(appliedAction);

    // ── open the next tick cohort ──────────────────────────────────────
    // Everything transmitted from here until the next decision belongs to
    // this cohort, tagged with the goal macro Python declared for it.
    g_currentMacroId = receivedMacroId;
    g_currentMacroTick = receivedMacroTick;
    {
        TvtTickCohort cohort;
        cohort.serial = g_nextCohortSerial++;
        cohort.macroId = g_currentMacroId;
        cohort.macroTick = g_currentMacroTick;
        cohort.action = appliedAction;
        cohort.openTimeSec = now;
        g_tickCohorts[cohort.serial] = cohort;
        g_openCohortSerial = cohort.serial;
        g_haveOpenCohort = true;
        ++g_cohortsOpened;
    }

    // LEGACY_ONLY telemetry anchor: duplication degree 가 유지된 결정 시각.
    // Scenario2 baseline 계측에만 쓰이며 state/reward/action 에는 관여하지 않는다.
    if (appliedAction == g_dqnState.currentAction) {
        g_lastNonHoDecisionSimTime = now;
    }

    NS_LOG_INFO("[V2xState@" << std::fixed << std::setprecision(2) << now << "s]"
        << "  uu_rssi="     << std::setprecision(1) << g_dqnState.uu_rssi  << "dBm"
        << "  pc5_rssi="    << g_dqnState.pc5_rssi_avg << "dBm"
        << "  current_action=" << TvtActionName(g_dqnState.currentAction)
        << "  action="      << appliedAction << "(" << TvtActionName(appliedAction) << ")");

    std::cout << "[DDQN@" << std::fixed << std::setprecision(2) << now << "s]"
              << "  uu_rssi="     << std::setprecision(1) << g_dqnState.uu_rssi    << "dBm"
              << "  pc5_rssi="    << g_dqnState.pc5_rssi_avg   << "dBm"
              << "  current_action=" << TvtActionName(g_dqnState.currentAction)
              << "  decision_id=" << currentDecisionId
              << "  requested=" << requestedAction
              << "  applied=" << appliedAction
              << "  status=" << AiApplyStatusName(g_lastAiResult.status)
              << "  macro=" << (receivedMacroId == TVT_AI_NO_MACRO
                                ? std::string("-")
                                : std::to_string(receivedMacroId))
              << "." << receivedMacroTick
              << "  cohort_backlog=" << g_tickCohorts.size()
              << std::endl;

    // 시뮬레이션 종료 100ms 전부터는 재스케줄링 중단.
    // done=1 은 Simulator::Run() 반환 후 SendAiDone() 이 단독으로 전송한다.
    if (now < g_simEndTime - 0.1)
        Simulator::Schedule(MilliSeconds(static_cast<uint32_t>(g_dqnDecisionIntervalMs)),
                             &CollectV2xState);
}

// ============================================================
// [J] SL 스케줄링/수신 DB 콜백 (기존 유지)
// ============================================================
void
NotifySlPscchScheduling(UeMacPscchTxOutputStats* pscchStats,
                        const SlPscchUeMacStatParameters pscchStatsParams)
{ pscchStats->Save(pscchStatsParams); }

void
NotifySlPsschScheduling(const SlPsschUeMacStatParameters psschStatsParams)
{
    // [수정] Save() 호출 제거 — 원래 psschStats->Save(psschStatsParams)로
    // SQLite DB에 기록하려 했으나, g_psschTxStats가 SetDb()로 DB에 연결된
    // 적이 없어 내부적으로 DB 쓰기를 시도하다 SIGSEGV 크래시가 발생했다
    // (NrSlUeMac::DoNrSlSlotIndication → NotifySlPsschScheduling → Save →
    //  WriteCache → SQLiteOutput::SpinExec). 이 프로젝트는 DB를 실제로
    // 활용하지 않으므로, DB 연결을 추가하는 대신 Save() 호출 자체를 제거하고
    // 순수 HARQ 재전송 카운팅만 수행한다.

    // HARQ 재전송 추적: NDI=0 → 재전송 (NDI=1 → 신규 데이터)
    // TS 38.321 §5.22.1: NR SL HARQ process ID reuse without NDI toggle = reTx
    g_slHarqTxCount++;
    if (psschStatsParams.ndi == 0)
        g_slHarqReTxCount++;
}

void
NotifySlPscchRx(UePhyPscchRxOutputStats* pscchStats,
                const SlRxCtrlPacketTraceParams pscchStatsParams)
{ pscchStats->Save(pscchStatsParams); }

void
NotifySlPsschRx(UePhyPsschRxOutputStats* psschStats,
                const SlRxDataPacketTraceParams psschStatsParams)
{ psschStats->Save(psschStatsParams); }

void
UePacketTraceDb(UeToUePktTxRxOutputStats* stats,
                Ptr<Node> node,
                const Address& localAddrs,
                std::string txRx,
                Ptr<const Packet> p,
                const Address& srcAddrs,
                const Address& dstAddrs,
                const SeqTsSizeHeader& seqTsSizeHeader)
{
    uint32_t nodeId  = node->GetId();
    uint64_t imsi    = node->GetDevice(0)->GetObject<NrUeNetDevice>()->GetImsi();
    uint32_t seq     = seqTsSizeHeader.GetSeq();
    uint32_t pktSize = p->GetSize() + seqTsSizeHeader.GetSerializedSize();
    stats->Save(txRx, localAddrs, nodeId, imsi, pktSize, srcAddrs, dstAddrs, seq);
}

void NotifySlRlcPduRx(UeRlcRxOutputStats* stats,
                      uint64_t imsi, uint16_t rnti, uint16_t txRnti,
                      uint8_t lcid, uint32_t rxPduSize, double delay)
{
    if (!stats) return;
    stats->Save(imsi, rnti, txRnti, lcid, rxPduSize, delay);
    // throughput/패킷 카운트는 App 소켓 콜백(NotifySlRxSizeDispatch)에서만 집계.
    // 이 trace는 15 UE × 3 BWP = 45개 콜백이 등록되어 SL 패킷 1개당 최대 45회 호출됨.
    // 여기서 g_totalRxBitsSl을 증가시키면 실제 대비 수십 배 과장되어 DDQN 상태 왜곡.
}

// ============================================================
// [K0] 비활성 경로 프로브 (100ms 주기)
// ============================================================
void ProbeBothPaths()
{
    double now = Simulator::Now().GetSeconds();
    if (now >= g_simEndTime - 0.1) return;

    // 두 경로를 모두 프로브한다. 현재 action 이 무엇이든 상관없다 — 그래야
    // 데이터를 싣지 않는 경로의 상태가 stale 해지지 않는다. 프로브는 QUIC
    // semantics 가 전혀 없는 순수 측정 패킷이다(연결 ID 없음, 패킷 번호 없음,
    // 핸드셰이크 없음).
    for (int pathId = 0; pathId < 2; ++pathId)
    {
        if (!g_pathTx.m_paths[pathId].valid || !g_pathTx.m_paths[pathId].sock)
            continue;

        uint32_t probeSeq = g_probeSeqNum++;
        Ptr<Packet> probe = Create<Packet>(16);
        SeqTsHeader seqHdr;
        seqHdr.SetSeq(probeSeq);
        probe->AddHeader(seqHdr);
        TvtMeasHeader meas;
        meas.type   = TVT_MEAS_PROBE;
        meas.pathId = static_cast<uint8_t>(pathId);
        probe->AddHeader(meas);

        if (g_pathTx.SendCopy(pathId, probe))
        {
            g_pendingProbes[{pathId, probeSeq}] = now;
        }
    }

    Simulator::Schedule(MilliSeconds(100), &ProbeBothPaths);
}

// ============================================================
// [K0b] 패킷 타임아웃 손실 감지 (100ms 주기)
// 200ms 초과 미확인 패킷 → 해당 경로 OnLoss() + 맵 정리
// ============================================================
void CheckPacketTimeout()
{
    double now = Simulator::Now().GetSeconds();
    // 판단주기의 kAckLossTimeoutMultiplier(10)배. 판단주기 기본값(100ms)
    // 기준 1.0s — 기존 값과 동일하다.
    const double kTimeoutSec = (g_dqnDecisionIntervalMs / 1000.0) * kAckLossTimeoutMultiplier;

    // logical packet ACK 타임아웃 (전체 손실 추정용, 경로 무관)
    for (auto it = g_pendingAcks.begin(); it != g_pendingAcks.end(); ) {
        if (now - it->second > kTimeoutSec) {
            g_ackOutcomeWindow.push_back(false);
            if (g_ackOutcomeWindow.size() > kAckLossWindowSize)
                g_ackOutcomeWindow.pop_front();
            it = g_pendingAcks.erase(it);
        } else {
            ++it;
        }
    }

    // 경로별 프로브 타임아웃 — 이것이 비활성 경로의 loss 표본을 만든다.
    for (auto it = g_pendingProbes.begin(); it != g_pendingProbes.end(); ) {
        if (now - it->second > kProbeTimeoutSec) {
            const int pathId = it->first.first;
            if (pathId >= 0 && pathId <= 1) g_pathLoss[pathId].Note(false);
            it = g_pendingProbes.erase(it);
        } else {
            ++it;
        }
    }

    if (now < g_simEndTime - 0.1)
        Simulator::Schedule(MilliSeconds(100), &CheckPacketTimeout);
}

// ============================================================
// [K1] ToD 주기 발신 (server DL 소켓, DL 방향)
// ============================================================
void SendTodPeriodic(Time interval)
{
    // 하나의 logical application packet. duplication 으로 두 copy 가 나가더라도
    // seq 는 하나뿐이며, 수신측은 이 seq 로 중복을 식별한다.
    uint32_t seq = g_todSeqNum++;
    Ptr<Packet> pkt = Create<Packet>(g_todPktSize);
    SeqTsSizeHeader hdr;
    hdr.SetSeq(seq);
    hdr.SetSize(g_todPktSize);
    pkt->AddHeader(hdr);

    int attempted = 0;
    bool duplicated = false;
    int successful = g_pathTx.SendLogicalPacket(pkt, &attempted, &duplicated);

    if (successful > 0) {
        double txTs = Simulator::Now().GetSeconds();
        g_txTodPackets++;
        ++g_logicalTxPackets;
        ++g_windowLogicalTx;
        // attempted 는 duplication degree 가 이 packet 에 대해 요구한 copy 수다.
        // 중복을 고른 뒤 한쪽 send 가 실패해도 resource cost 가 단일 copy 수준으로
        // 내려가면 안 되므로 reward 는 이 값을 쓴다. successful 은 진단용이다.
        g_attemptedCopiesTx      += static_cast<uint64_t>(attempted);
        g_windowAttemptedCopies  += static_cast<uint64_t>(attempted);
        g_successfulCopiesTx     += static_cast<uint64_t>(successful);
        g_windowSuccessfulCopies += static_cast<uint64_t>(successful);
        // action label 이 아니라 accumulator 가 이 packet 에 대해 실제로 내린
        // 판정을 센다. partial duplication 에서 둘은 다르다.
        if (duplicated) {
            ++g_duplicatedLogicalPackets;
            ++g_windowDuplicatedLogicalPackets;
        }
        // 이 logical packet 을 실제로 송신한 decision tick(cohort)에 귀속.
        // deadline 은 250 ms 뒤에 성숙하지만 결과는 여기로 돌아온다.
        TvtDeadlineRecord record;
        record.txTimeSec = txTs;
        record.deliveredInTime = false;
        record.deliveryDelayMs = -1.0;
        record.cohortSerial = g_openCohortSerial;
        record.hasCohort = g_haveOpenCohort;
        g_tvtDeadlinePending[seq] = record;
        if (g_haveOpenCohort)
        {
            auto cohortIt = g_tickCohorts.find(g_openCohortSerial);
            if (cohortIt != g_tickCohorts.end())
            {
                TvtTickCohort& cohort = cohortIt->second;
                ++cohort.logicalTx;
                ++cohort.pendingDeadlines;
                cohort.attemptedCopies += static_cast<uint64_t>(attempted);
                if (duplicated)
                {
                    ++cohort.duplicatedLogicalPackets;
                }
            }
        }
        if (!g_disableTodAck)
        {
            g_pendingAcks[seq] = txTs;
        }
    }

    double now = Simulator::Now().GetSeconds();
    if (now < g_simEndTime - 0.5)
        Simulator::Schedule(interval, &SendTodPeriodic, interval);
}

// ============================================================
// [K2] RSU Uu→SL 릴레이 콜백 (DL 방향)
// ============================================================
// PC5 모드에서 server가 RSU의 v2v0 IP로 ToD를 보내면,
// RSU가 Uu로 수신하여 SL을 통해 carNode(ue1V2v0Addr)로 포워딩한다.
// SeqTsSizeHeader는 원본 그대로 전달하여 carNode에서 E2E 지연 측정 가능.
void NotifyRsuRelayRx(Ptr<Socket> socket)
{
    if (g_enablePc5Debug)
    {
        std::cout << "[RELAY-RX-ENTRY] @ " << Simulator::Now().GetSeconds() << "s" << std::endl;
    }

    // 방어 코드: 콜백 인자/노드 포인터 유효성 확인
    if (!socket || !socket->GetNode()) return;

    auto mapIt = g_rsuRelaySocketMap.find(socket->GetNode()->GetId());
    if (mapIt == g_rsuRelaySocketMap.end()) {
        std::cout << "[RELAY-RX-NOMAP] nodeId=" << socket->GetNode()->GetId() << std::endl;
        return;
    }
    Ptr<Socket> txSock = mapIt->second;
    if (!txSock) return;  // 방어 코드: 매핑된 TX 소켓이 없으면 스킵

    Ptr<Packet> pkt;
    Address from;
    uint32_t maxDrain = 50;   // 한 이벤트에 최대 50개 처리
    uint32_t drained  = 0;

    while (drained++ < maxDrain && (pkt = socket->RecvFrom(from)))
    {

       Ptr<Packet> fwdPkt = pkt->Copy();

        int sent = txSock->Send(fwdPkt);

        if (sent > 0) {
            if (g_enablePc5Debug) {
                std::cout << "[RSU->PC5] forward "
                          << fwdPkt->GetSize()
                          << " bytes @ " << Simulator::Now().GetSeconds()
                          << "s" << std::endl;
            }
        } else {
            std::cout << "[RELAY-TX-FAIL] sent=" << sent
                      << " @ " << Simulator::Now().GetSeconds() << "s" << std::endl;
        }
    }
}
// ============================================================
// [K3] ToD RTT 루프백 — RemoteHost에서 ACK 수신 → RTT 계산
// ============================================================
// UE1이 ToD 패킷 수신 즉시 ACK(64B + SeqTsHeader)를 RemoteHost로 보내고,
// RemoteHost가 수신 시 g_pendingAcks에 저장된 송신 시각으로 RTT를 계산한다.
void NotifyTodAckRx(Ptr<Socket> socket)
{
    Ptr<Packet> pkt;
    Address from;
    while ((pkt = socket->RecvFrom(from))) {
        if (!pkt || pkt->GetSize() < 1) continue;

        Ptr<Packet> copy = pkt->Copy();
        TvtMeasHeader meas;
        if (copy->GetSize() < meas.GetSerializedSize()) continue;
        copy->RemoveHeader(meas);

        SeqTsHeader seqHdr;
        if (copy->GetSize() < seqHdr.GetSerializedSize()) continue;
        copy->PeekHeader(seqHdr);
        const uint32_t seq  = seqHdr.GetSeq();
        const int      path = static_cast<int>(meas.pathId);
        if (path < 0 || path > 1) continue;   // 역직렬화 값은 신뢰하지 않는다

        const double now = Simulator::Now().GetSeconds();

        // ── 측정 프로브 응답: 경로별 RTT/생존 확인 ────────────────────
        if (meas.type == TVT_MEAS_PROBE_RESP) {
            auto it = g_pendingProbes.find({path, seq});
            if (it != g_pendingProbes.end()) {
                double rtt_ms = (now - it->second) * 1000.0;
                if (rtt_ms > 0.0 && rtt_ms < 10000.0) {
                    // 데이터가 흐르지 않는 경로에서는 이 왕복 측정의 절반이
                    // 유일한 실측 편도 지연 추정치다. 데이터가 흐르는 경로는
                    // DeliverTodPacket() 의 실측 편도 지연이 EMA 를 지배한다.
                    g_pathMeas[path].NoteDelaySample(rtt_ms / 2.0, now);
                    g_pathLoss[path].Note(true);
                }
                g_pendingProbes.erase(it);
            }
            continue;
        }

        // ── logical packet ACK: 전체 RTT/손실 표본 ────────────────────
        if (meas.type == TVT_MEAS_ACK) {
            auto it = g_pendingAcks.find(seq);
            if (it != g_pendingAcks.end()) {
                double rtt_ms = (now - it->second) * 1000.0;
                if (rtt_ms > 0.0 && rtt_ms < 10000.0) {
                    g_recentRtts.push_back(rtt_ms);
                    if (g_recentRtts.size() > kRttWindowSize)
                        g_recentRtts.pop_front();
                    g_lastAckDelayMs = rtt_ms;
                    g_ackOutcomeWindow.push_back(true);
                    if (g_ackOutcomeWindow.size() > kAckLossWindowSize)
                        g_ackOutcomeWindow.pop_front();
                }
                g_pendingAcks.erase(it);
            }
            continue;
        }
    }
}

// ============================================================
static void PrintFinalSummary()
{
    double totalTime = Simulator::Now().GetSeconds();
    if (totalTime <= 0.0) return;

    double avgDelayMs = (g_delayCount > 0) ? g_delaySumMs / static_cast<double>(g_delayCount) : 0.0;
    double lossRateFinal = (g_txTodPackets > 0)
        ? std::max(0.0, 1.0 - static_cast<double>(g_rxTodPackets) / static_cast<double>(g_txTodPackets))
        : 0.0;

    std::cout << "\n======================================================" << std::endl;
    std::cout << "   SIMULATION COMPLETE: FINAL SYSTEM PERFORMANCE   "     << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << " • Total Simulation Time : "
              << std::fixed << std::setprecision(2) << totalTime << " sec"  << std::endl;
    std::cout << " • Avg E2E Delay          : "
              << std::setprecision(3) << avgDelayMs << " ms" << std::endl;
    std::cout << " • Max E2E Delay          : "
              << std::setprecision(3) << g_delayMaxMs << " ms" << std::endl;
    std::cout << " • Packet Loss Rate       : "
              << std::setprecision(3) << (lossRateFinal * 100.0) << " %" << std::endl;
    std::cout << " • Action switch count    : " << g_switchCount << std::endl;
    std::cout << "======================================================" << std::endl;
}

// ============================================================
// [K7] 처리량 주기 출력 (1초)
// ============================================================
void PrintThroughput()
{
    double time = Simulator::Now().GetSeconds();

    double recentAvgDelayMs = 0.0;
    if (!g_recentE2eDelaysMs.empty())
    {
        double sum = 0.0;
        for (double d : g_recentE2eDelaysMs) sum += d;
        recentAvgDelayMs = sum / static_cast<double>(g_recentE2eDelaysMs.size());
    }

    double cumulativeLossRate = (g_txTodPackets > 0)
        ? std::max(0.0, 1.0 - static_cast<double>(g_rxTodPackets) / static_cast<double>(g_txTodPackets))
        : 0.0;

    std::cout << "[Status @ " << std::fixed << std::setprecision(1) << time << "s] "
              << "avg_delay=" << std::setprecision(2) << recentAvgDelayMs << "ms"
              << "  cum_loss=" << std::setprecision(3) << (cumulativeLossRate * 100.0) << "%"
              << "  current_action=" << TvtActionName(g_currentAction)
              << "  action_switches=" << g_switchCount
              << "  uu_rssi=" << std::setprecision(1) << g_lastUuRssi << "dBm"
              << "  pc5_rssi=" << g_lastPc5Rssi << "dBm"
              << std::endl;

    if (time < Simulator::GetMaximumSimulationTime().GetSeconds() - 0.5)
        Simulator::Schedule(Seconds(1.0), &PrintThroughput);
}

// ============================================================
// [M] Seamless 수직 핸드오버 판단 함수 (100ms 주기)
// ============================================================
// [K] PC5 경로 RSU 동적 선택
// ============================================================
uint32_t FindNearestRsu(NodeContainer& rsuNodes, Ptr<Node> carNode)
{
    Ptr<MobilityModel> carMob = carNode->GetObject<MobilityModel>();
    if (!carMob) return 0;
    Vector carPos = carMob->GetPosition();
    uint32_t bestIdx  = 0;
    double   bestDist = 1e18;
    for (uint32_t i = 0; i < rsuNodes.GetN(); ++i) {
        Ptr<MobilityModel> rsuMob = rsuNodes.Get(i)->GetObject<MobilityModel>();
        if (!rsuMob) continue;
        Vector rsuPos = rsuMob->GetPosition();
        double dx   = carPos.x - rsuPos.x;
        double dy   = carPos.y - rsuPos.y;
        double dist = std::sqrt(dx*dx + dy*dy);
        if (dist < bestDist) { bestDist = dist; bestIdx = i; }
    }
    return bestIdx;
}

void UpdatePc5Path()
{
    double now = Simulator::Now().GetSeconds();
    if (now >= g_simEndTime - 0.1) return;
    if (!g_carNodePtr) { Simulator::Schedule(MilliSeconds(100), &UpdatePc5Path); return; }

    uint32_t nearestIdx = FindNearestRsu(g_rsuNodesGlobal, g_carNodePtr);
    if (nearestIdx != g_activePc5RsuIdx &&
        nearestIdx < g_rsuBackhaulAddrs.size())
    {
        g_activePc5RsuIdx = nearestIdx;
        g_pathTx.m_paths[1].dst =
            InetSocketAddress(g_rsuBackhaulAddrs[nearestIdx], g_uuPortGlobal);
        std::cout << "[PC5-RSU] 활성 RSU 변경 → RSU" << nearestIdx
                  << " backhaul=" << g_rsuBackhaulAddrs[nearestIdx]
                  << " @ " << std::fixed << std::setprecision(2)
                  << now << "s" << std::endl;
    }
    Simulator::Schedule(MilliSeconds(100), &UpdatePc5Path);
}

// ============================================================
void CheckAndSwitchInterface()
{
    double now = Simulator::Now().GetSeconds();
    if (now >= g_simEndTime - 0.2) return;

    // [RSU 거리 계측] 차량-RSU 최소 거리(전체 런 기준) 갱신 — enablePc5Debug와
    // 무관하게 항상 추적한다(2.4/2.9 최종 출력 요구사항).
    if (g_carNodePtr && g_rsuNodesGlobal.GetN() > 0)
    {
        uint32_t nearIdx = FindNearestRsu(g_rsuNodesGlobal, g_carNodePtr);
        Ptr<MobilityModel> rsuMob = g_rsuNodesGlobal.Get(nearIdx)->GetObject<MobilityModel>();
        Ptr<MobilityModel> carMob = g_carNodePtr->GetObject<MobilityModel>();
        if (rsuMob && carMob)
        {
            double d = CalculateDistance(rsuMob->GetPosition(), carMob->GetPosition());
            if (d < g_minRsuCarDist)
            {
                g_minRsuCarDist     = d;
                g_minRsuCarDistTime = now;
            }
        }
    }

    // duplication degree 가 달라지면 여기서 적용한다. 5개 level 사이의 모든
    // 전이를 동일하게 센다. 이것은 물리 handover 가 아니다: 연결은 어느
    // 경우에도 재수립되지 않고, 정수 하나가 바뀔 뿐이다.
    // (아래 g_switchCount 는 그래서 "duplication-level change" 진단 카운터다.)
    int newAction = g_pathTx.GetAction();

    // No cooldown and no hold: the transition costs the modelled network
    // nothing, so nothing gates it.
    if (newAction != g_currentAction)
    {
        std::string from = std::string(TvtActionName(g_currentAction)) + " → "
                           + TvtActionName(newAction);

        // [Time on level] 직전 duplication level 에 경과시간 누적.
        double elapsed = now - g_lastIfChangeAccountTime;
        if (g_currentAction >= 0 && g_currentAction < TVT_AI_ACTION_COUNT)
            g_timeOnLevelSec[g_currentAction] += elapsed;
        g_lastIfChangeAccountTime = now;

        const int prevAction = g_currentAction;
        g_currentAction = newAction;
        g_switchCount++;
        if (g_pendingAiDecision && newAction == g_pendingAiRequestedAction)
        {
            g_lastAiResult = {g_pendingAiDecisionId,
                              g_pendingAiRequestedAction,
                              newAction,
                              TVT_AI_APPLY_APPLIED,
                              true};
            std::cout << "[ACTION-APPLY] decision_id=" << g_pendingAiDecisionId
                      << " requested=" << g_pendingAiRequestedAction
                      << " applied=" << newAction
                      << " status=APPLIED sim_time_s=" << now << std::endl;
            g_pendingAiDecision = false;
        }
        // CORE: env 계약의 time_since_last_switch_ms 앵커. gate 하지 않는다.
        g_lastActionSwitchTimeSec = now;
        // LEGACY_ONLY: 아래 두 항목은 duplication-level 변경 진단용이며 물리
        // handover 가 아니다.
        if (g_enableLegacyHoTelemetry)
        {
            g_interruptionActive = true;
            g_interruptionStartSimTime = now;
            g_hoSwitchTimes.push_back(
                {now, std::string(TvtActionName(prevAction)) + "2"
                      + TvtActionName(newAction)});
        }
        std::cout << "\n[ACTION-SWITCH] " << from << " @ "
                  << std::fixed << std::setprecision(2) << now << "s"
                  << "  uu_rssi=" << std::setprecision(1) << g_lastUuRssi << "dBm"
                  << "  pc5_rssi=" << g_lastPc5Rssi << "dBm\n"
                  << std::endl;

        // ── 시나리오 측정 기록 (LEGACY_ONLY, 기본 OFF) ────────────────────
        if (g_enableLegacyHoTelemetry && g_pendingHoMeas) {
            g_hoMeasIdx++;
            double switchWall = ScenarioWallNow();

            // 시뮬레이션 시간 기준 Interface Switching 구간 (D1 → D2).
            // CheckAndSwitchInterface()의 100ms 폴링 구조에 정확히 대응하는
            // 값으로, 항상 유한(대략 0~110ms)하다 — wall-clock이 호스트
            // 스케줄링 잡음으로 오염됐을 때의 안전한 대체값(fallback)으로 사용.
            double simSwitchDelay = (now - g_pend_s2_D1) * 1000.0;

            // Scenario 1: DDQN Inference Delay + Interface Switching Delay (wall-clock)
            double s1_dqnInferDelay  = (g_pend_s3_actionWall - g_pend_s1_inferWall) * 1000.0;
            double s1_ifSwitchDelay  = (switchWall           - g_pend_s3_actionWall) * 1000.0;
            double s1_total          = s1_dqnInferDelay + s1_ifSwitchDelay;
            if (s1_total > kMaxPlausibleDelayMs) {
                // 다른 무관한 10ms 틱의 Python IPC 왕복(실제 CPU 처리 시간)이
                // 이 폴링 구간 사이에 끼어들어 오염된 것으로 판단 — 동일 구간의
                // 시뮬레이션 시간 값으로 대체(핸드오버/DDQN 로직은 변경 없음).
                std::cout << "[Scenario1][HO-" << g_hoMeasIdx << "] 측정 이상치 감지"
                          << " (raw=" << std::fixed << std::setprecision(1) << s1_total
                          << "ms > " << kMaxPlausibleDelayMs << "ms) → sim-time 값으로 대체\n";
                s1_ifSwitchDelay = simSwitchDelay;
                s1_total         = s1_dqnInferDelay + s1_ifSwitchDelay;
            }
            std::cout << "[Scenario1][HO-" << g_hoMeasIdx << "]"
                      << "  dqn_infer="   << std::fixed << std::setprecision(3) << s1_dqnInferDelay << "ms"
                      << "  if_switch="   << s1_ifSwitchDelay << "ms"
                      << "  total="       << s1_total << "ms\n";
            if (g_csvS1.is_open())
                g_csvS1 << g_hoMeasIdx << ","
                        << std::fixed << std::setprecision(4)
                        << s1_dqnInferDelay << "," << s1_ifSwitchDelay << "," << s1_total << "\n";

            // Scenario 2: 핸드오버 구간(D1→D2) vs 평상시 baseline(다음 폴링까지)의 평균 비교
            double s2_avgHoDelay    = simSwitchDelay;
            double s2_avgNonHoDelay = (g_curNonHoDelayCount > 0)
                                        ? g_curNonHoDelaySum / g_curNonHoDelayCount : 0.0;
            double s2_diff          = s2_avgHoDelay - s2_avgNonHoDelay;
            std::cout << "[Scenario2][HO-" << g_hoMeasIdx << "]"
                      << "  avg_ho_delay="     << std::fixed << std::setprecision(3) << s2_avgHoDelay << "ms"
                      << "  avg_non_ho_delay=" << s2_avgNonHoDelay << "ms"
                      << " (n="  << g_curNonHoDelayCount << ")"
                      << "  diff=" << s2_diff << "ms\n";
            if (g_csvS2.is_open())
                g_csvS2 << g_hoMeasIdx << ","
                        << std::fixed << std::setprecision(4)
                        << s2_avgHoDelay << "," << s2_avgNonHoDelay << "," << s2_diff << "\n";
            g_grandHoDelaySum += s2_avgHoDelay; ++g_grandHoDelayCount;
            // 다음 HO 구간까지의 평상시 baseline 집계를 위해 리셋 (전체 누적치는 유지)
            g_curNonHoDelaySum = 0.0; g_curNonHoDelayCount = 0;

            // Scenario 3: CppSendBegin 직전 → switch_complete
            //   Delay1 = ns-3→Python IPC + Python wakeup (wall-clock, 단일 이벤트 내)
            //   Delay2 = DDQN inference (numpy + torch + action 생성, wall-clock, 단일 이벤트 내)
            //   Delay3 = action 반환 + Interface Switching (wall-clock, 이상치 시 sim-time 대체)
            double s3_d1    = (g_pend_s3_inferWall  - g_pend_s3_sendWall)  * 1000.0;
            double s3_d2    = (g_pend_s3_actionWall  - g_pend_s3_inferWall) * 1000.0;
            double s3_d3    = (switchWall            - g_pend_s3_actionWall) * 1000.0;
            double s3_total = s3_d1 + s3_d2 + s3_d3;
            if (s3_total > kMaxPlausibleDelayMs) {
                std::cout << "[Scenario3][HO-" << (g_hoMeasIdx) << "] 측정 이상치 감지"
                          << " (raw=" << std::fixed << std::setprecision(1) << s3_total
                          << "ms > " << kMaxPlausibleDelayMs << "ms) → sim-time 값으로 대체\n";
                s3_d3    = simSwitchDelay;
                s3_total = s3_d1 + s3_d2 + s3_d3;
            }
            std::cout << "[Scenario3][HO-" << g_hoMeasIdx << "]\n"
                      << "  Delay1 (ns3->Python) = "
                      << std::fixed << std::setprecision(3) << s3_d1    << " ms\n"
                      << "  Delay2 (DDQN)         = " << s3_d2    << " ms\n"
                      << "  Delay3 (Switch)      = " << s3_d3    << " ms\n"
                      << "  Total                = " << s3_total << " ms\n";
            if (g_csvS3.is_open())
                g_csvS3 << g_hoMeasIdx << ","
                        << std::fixed << std::setprecision(4)
                        << s3_d1 << "," << s3_d2 << "," << s3_d3 << "," << s3_total << "\n";

            g_pendingHoMeas = false;
        }
    }
    else if (g_enableLegacyHoTelemetry && !g_pendingHoMeas &&
             g_lastNonHoDecisionSimTime >= 0.0)
    {
        // [Scenario2] LEGACY_ONLY baseline 샘플: 가장 최근 유지
        // (action==currentAction) 결정 시점부터 이번 폴링 시점까지의 간격 —
        // Scenario2의 Handover Delay와 동일한 측정 기준(시뮬레이션 시간,
        // 100ms 폴링 구조)으로 비교 가능하다.
        double sample = (now - g_lastNonHoDecisionSimTime) * 1000.0;
        if (sample >= 0.0) {
            g_curNonHoDelaySum   += sample; ++g_curNonHoDelayCount;
            g_grandNonHoDelaySum += sample; ++g_grandNonHoDelayCount;
        }
    }

    Simulator::Schedule(MilliSeconds(100), &CheckAndSwitchInterface);
}

// ============================================================
// T3 — temporal load schedule
// ============================================================
// active vehicle 의 CAM 송신을 켜고 끈다. 노드를 생성/삭제하지 않는다:
// 바꾸려는 것은 topology 가 아니라 network load 다.
//
// 전이 시각은 공칭값에 rngRun 유도 jitter(+/- kLoadJitterSec)를 더한 값이다.
// 같은 master seed + episode -> 동일 전이, 다른 episode -> 다른 전이.
static void ToggleCamLoad(size_t index)
{
    g_camEnabled = !g_camEnabled;
    std::cout << ">>> [LOAD-PHASE] t=" << std::fixed << std::setprecision(3)
              << Simulator::Now().GetSeconds() << "s  transition=" << index
              << "  cam_enabled=" << (g_camEnabled ? 1 : 0) << std::endl;
}

static void ScheduleCamLoadPhases(double simEndSec, bool scheduleToggles)
{
    // DOMAIN_D. 전용 stream 이고, scheduleToggles 와 무관하게 항상 같은 수만큼
    // 뽑는다 — L0/L1/L2 가 동일한 전이 시각을 갖게 하기 위해서다. numActiveVeh
    // 는 이 stream 에 전혀 영향을 주지 않는다.
    Ptr<UniformRandomVariable> uJit =
        CreateObjectWithAttributes<UniformRandomVariable>(
            "Stream", IntegerValue(kStreamLoadJitter),
            "Min", DoubleValue(-kLoadJitterSec),
            "Max", DoubleValue(kLoadJitterSec));

    // 시작은 저부하: 첫 전이에서 켜진다.
    g_camEnabled = false;
    g_loadTransitions.clear();
    double prev = 0.0;
    for (size_t i = 0; i < g_loadNominalSec.size(); ++i)
    {
        double t = g_loadNominalSec[i] + uJit->GetValue();
        // 단조성과 범위를 보장한다. jitter 가 순서를 뒤집으면 phase 구조가
        // 무너지므로 조용히 통과시키지 않고 바로잡는다.
        t = std::max(t, prev + 0.5);
        t = std::min(t, simEndSec - 0.5);
        g_loadTransitions.push_back(t);
        prev = t;
        if (scheduleToggles) Simulator::Schedule(Seconds(t), &ToggleCamLoad, i);
    }
    std::cout << ">>> [LOAD-SCHEDULE] nominal=";
    for (size_t i = 0; i < g_loadNominalSec.size(); ++i)
        std::cout << g_loadNominalSec[i]
                  << (i + 1 < g_loadNominalSec.size() ? "," : "");
    std::cout << "  jittered=" << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < g_loadTransitions.size(); ++i)
        std::cout << g_loadTransitions[i] << (i + 1 < g_loadTransitions.size() ? "," : "");
    std::cout << "  jitter_range=+/-" << kLoadJitterSec << "s"
              << "  start_state=low" << std::endl;
}

// ============================================================
// Mobility window offset
// ============================================================
// Ns2MobilityHelper 는 시간 오프셋을 지원하지 않으므로, trace 를 한 번
// 재작성해 window 의 시작을 t=0 으로 옮긴다.
//
//   - t < offset 인 setdest 는 버린다
//   - 남은 setdest 의 시각에서 offset 을 뺀다
//   - 초기 X_/Y_/Z_ 를 offset 시점의 waypoint 로 바꾼다
//
// 궤적 자체는 바뀌지 않는다. 같은 도로의 다른 구간을 episode 로 쓰는 것뿐이다.
// 기존 window 를 쓰려면 offset 0 을 주면 된다(파일을 그대로 사용).
static std::string
BuildOffsetMobilityTrace(const std::string& src, double offsetSec,
                         const std::string& outDir)
{
    if (offsetSec <= 0.0) return src;

    std::ifstream in(src);
    NS_ABORT_MSG_IF(!in.is_open(), "cannot open mobility trace: " << src);

    std::vector<std::pair<double, std::string>> moves;  // (t, "x y speed")
    std::string line;
    std::string firstX, firstY, firstZ;
    while (std::getline(in, line))
    {
        // "$ns_ at <t> \"$node_(0) setdest <x> <y> <spd>\""
        const auto atPos = line.find("$ns_ at ");
        if (atPos != std::string::npos)
        {
            std::istringstream ss(line.substr(atPos + 8));
            double t;
            if (!(ss >> t)) continue;
            const auto sdPos = line.find("setdest ");
            if (sdPos == std::string::npos) continue;
            std::string rest = line.substr(sdPos + 8);
            const auto q = rest.rfind('"');
            if (q != std::string::npos) rest = rest.substr(0, q);
            moves.emplace_back(t, rest);
            continue;
        }
        if (line.find(" set X_ ") != std::string::npos) firstX = line;
        else if (line.find(" set Y_ ") != std::string::npos) firstY = line;
        else if (line.find(" set Z_ ") != std::string::npos) firstZ = line;
    }
    in.close();
    NS_ABORT_MSG_IF(moves.empty(), "mobility trace has no setdest lines: " << src);

    // offset 시점의 waypoint 를 초기 위치로 삼는다.
    size_t startIdx = 0;
    while (startIdx + 1 < moves.size() && moves[startIdx + 1].first <= offsetSec)
        ++startIdx;
    NS_ABORT_MSG_IF(moves.back().first <= offsetSec,
                    "mobilityStartOffsetS=" << offsetSec
                    << " is beyond the trace end " << moves.back().first);

    double sx = 0.0, sy = 0.0, sspd = 0.0;
    {
        std::istringstream ss(moves[startIdx].second);
        ss >> sx >> sy >> sspd;
    }

    const std::string outPath = outDir + "tvt_mobility_window.tcl";
    std::ofstream out(outPath);
    NS_ABORT_MSG_IF(!out.is_open(), "cannot write offset trace: " << outPath);
    out << std::fixed << std::setprecision(6);
    out << "$node_(0) set X_ " << sx << "\n";
    out << "$node_(0) set Y_ " << sy << "\n";
    out << "$node_(0) set Z_ 0\n";
    size_t emitted = 0;
    for (size_t i = startIdx; i < moves.size(); ++i)
    {
        const double t = moves[i].first - offsetSec;
        if (t < 0.0) continue;
        out << "$ns_ at " << t << " \"$node_(0) setdest "
            << moves[i].second << "\"\n";
        ++emitted;
    }
    out.close();
    std::cout << ">>> [MOBILITY-WINDOW] offset=" << offsetSec << "s"
              << "  start=(" << sx << "," << sy << ")"
              << "  waypoints=" << emitted
              << "  src=" << src << "  -> " << outPath << std::endl;
    return outPath;
}

// ============================================================
// [O] gnuplot / 위치 기록 함수 (기존 유지)
// ============================================================
void
WriteInitPosGnuScript(std::string posFilename)
{
    std::ofstream outFile;
    std::string filename = "gnu-script-" + posFilename;
    outFile.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);
    if (!outFile.is_open()) { NS_LOG_ERROR("Can't open file " << filename); return; }
    std::string pngFileName = posFilename.substr(0, posFilename.rfind('.'));
    outFile << "set terminal png" << std::endl;
    outFile << "set output \"" << pngFileName << ".png\"" << std::endl;
    outFile << "set style line 1 lc rgb 'black' ps 2 pt 7" << std::endl;
    outFile << "unset key" << std::endl;
    outFile << "set grid" << std::endl;
    outFile << "plot \"" << posFilename << "\" using 3:4 with points ls 1";
    outFile.close();
}

void
PrintUeInitPosToFile(std::string filename)
{
    std::ofstream outFile;
    outFile.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);
    if (!outFile.is_open()) { NS_LOG_ERROR("Can't open file " << filename); return; }
    for (NodeList::Iterator it = NodeList::Begin(); it != NodeList::End(); ++it) {
        Ptr<Node> node = *it;
        for (int j = 0; j < (int)node->GetNDevices(); j++) {
            Ptr<NrUeNetDevice> uedev = node->GetDevice(j)->GetObject<NrUeNetDevice>();
            if (uedev) {
                Vector pos = node->GetObject<MobilityModel>()->GetPosition();
                outFile << node->GetId() << " " << uedev->GetImsi()
                        << " " << pos.x << " " << pos.y << std::endl;
            }
        }
    }
    WriteInitPosGnuScript(filename);
}

void
RecordMobility(bool FirstWrite, std::string fileName)
{
    std::ofstream outFile;
    if (FirstWrite) {
        outFile.open(fileName.c_str(), std::ios_base::out);
    } else {
        outFile.open(fileName.c_str(), std::ios_base::app);
        outFile << std::endl << std::endl;
    }
    for (NodeList::Iterator it = NodeList::Begin(); it != NodeList::End(); ++it) {
        Ptr<Node> node = *it;
        for (int j = 0; j < (int)node->GetNDevices(); j++) {
            Ptr<NrUeNetDevice> uedev = node->GetDevice(j)->GetObject<NrUeNetDevice>();
            if (uedev) {
                Vector pos = node->GetObject<MobilityModel>()->GetPosition();
                outFile << Simulator::Now().GetSeconds() << " " << node->GetId()
                        << " " << uedev->GetImsi() << " " << pos.x << " " << pos.y
                        << std::endl;
            }
        }
    }
    Simulator::Schedule(Seconds(1), &RecordMobility, false, fileName);
}

void
WriteGifGnuScript(std::string MobilityFileName,
                  Time simTime, double speed,
                  Ptr<Node> firstUeNode, Ptr<Node> lastUeNode)
{
    std::ofstream outFile;
    std::string fileName = "gif-script-" + MobilityFileName;
    outFile.open(fileName.c_str(), std::ios_base::out | std::ios_base::trunc);
    if (!outFile.is_open()) { NS_LOG_ERROR("Can't open file " << fileName); return; }
    outFile << "set term gif animate delay 100" << std::endl;
    std::string gifFileName = MobilityFileName.substr(0, MobilityFileName.rfind('.'));
    outFile << "set output \"" << gifFileName << ".gif\"" << std::endl;
    outFile << "unset key" << std::endl;
    outFile << "set grid" << std::endl;
    Vector firstNodePos = firstUeNode->GetObject<MobilityModel>()->GetPosition();
    Vector LastNodePos  = lastUeNode->GetObject<MobilityModel>()->GetPosition();
    outFile << "set xrange [" << (firstNodePos.x - 10.0) << ":"
            << (simTime.GetSeconds() * speed + LastNodePos.x) << "]" << std::endl;
    outFile << "set yrange [" << (firstNodePos.y - 10.0) << ":"
            << (LastNodePos.y + 10.0) << "]" << std::endl;
    outFile << "do for [i=0:" << simTime.GetSeconds() - 1 << "] {plot \""
            << MobilityFileName << "\" index i using 4:5}" << std::endl;
}

void
GetSlBitmapFromString(std::string slBitMapString, std::vector<std::bitset<1>>& slBitMapVector)
{
    static std::unordered_map<std::string, uint8_t> lookupTable = {{"0", 0}, {"1", 1}};
    std::stringstream ss(slBitMapString);
    std::string token;
    std::vector<std::string> extracted;
    while (std::getline(ss, token, '|')) extracted.push_back(token);
    for (const auto& v : extracted) {
        if (lookupTable.find(v) == lookupTable.end())
            NS_FATAL_ERROR("Bit type " << v << " not valid. Valid values: 0 and 1");
        slBitMapVector.emplace_back(lookupTable[v] & 0x01);
    }
}

void
SavePositionPerIP(V2xKpi* v2xKpi)
{
    for (NodeList::Iterator it = NodeList::Begin(); it != NodeList::End(); ++it) {
        Ptr<Node> node = *it;
        for (int j = 0; j < (int)node->GetNDevices(); j++) {
            Ptr<NrUeNetDevice> uedev = node->GetDevice(j)->GetObject<NrUeNetDevice>();
            if (uedev) {
                Ptr<Ipv4L3Protocol> ipv4Protocol = node->GetObject<Ipv4L3Protocol>();
                Ipv4InterfaceAddress addresses = ipv4Protocol->GetAddress(1, 0);
                std::ostringstream ueIpv4Addr;
                ueIpv4Addr << addresses.GetLocal();
                Vector pos = node->GetObject<MobilityModel>()->GetPosition();
                v2xKpi->FillPosPerIpMap(ueIpv4Addr.str(), pos);
            }
        }
    }
}

// ============================================================
// [P] main()
// ============================================================
int
main(int argc, char* argv[])
{
    // crash backtrace handler
    signal(SIGABRT, [](int) {
        void* buf[64];
        int n = backtrace(buf, 64);
        backtrace_symbols_fd(buf, n, STDERR_FILENO);
        _exit(1);
    });
    signal(SIGSEGV, [](int) {
        void* buf[64];
        int n = backtrace(buf, 64);
        backtrace_symbols_fd(buf, n, STDERR_FILENO);
        _exit(1);
    });

    Ipv4StaticRoutingHelper ipv4RoutingHelper;

    // ----------------------------------------------------------
    // 파라미터 정의
    // ----------------------------------------------------------
    // NOTE: 차량 배치는 SUMO trace(Ns2MobilityHelper)가 담당한다. 격자 배치용
    // numVehiclesPerLane/numLanes/interVehicleDist/interLaneDist/
    // enableOneTxPerLane 은 유일한 소비자였던 InstallHighwayMobility() 와 함께
    // 제거했다. speed 는 gnuplot GIF 스크립트 생성기가 계속 사용한다.
    double   speed              = 38.88889; // m/s (≈140 km/h)
    bool     logging            = false;
    // PC5 실측 손실률(84%) 분석 결과 HARQ 미재전송이 주요 원인 중 하나로
    // 확인되어 true로 변경(7월 15일에 false로 수정됐던 것을 되돌림).
    // SetSlPsfchPeriod()도 이에 맞춰 0이 아닌 값으로 함께 조정 필요(아래 참조).
    bool     harqEnabled        = true;
    Time     delayBudget        = MilliSeconds(200);

    bool     useIPv6            = false;
    uint32_t udpPacketSizeBe    = 200;
    double   dataRateBe         = 16;       // kbps

    // vehicle_mobility.tcl 분석: 마지막 setdest는 417.0s, 차량은 413.0s에
    // 출발점(5104.99, 6694.51)으로부터 2.22m 이내까지 복귀 — 왕복 1회 =
    // 에피소드 1회이므로 그 직후(414~417s는 출발점을 지나 계속 이동하는
    // 잉여 구간)인 415.0s를 기본 종료 시각으로 사용한다.
    Time simTime                  = Seconds(415); // 왕복 1회 = 에피소드 1회
    Time slBearersActivationTime  = MilliSeconds(10);  // RRC 연결 전 활성화

    // ── 서비스별 주파수 대역 설정 (3GPP 표준 기반) ──────────────────────
    // 서비스 1: 원격제어 (Remote Control, port 1234, ToD 영상)
    //   인터페이스: Uu/wwan0 (5G gNB 경유)
    //   주파수 대역: 3.5 GHz, n78 (Sub-6 GHz eMBB, 핸드오버 대상)
    double freqRemoteControl   = 3.5e9;   // [Hz] 3GPP n78
    double bwRemoteControl     = 100e6;   // [Hz] 100 MHz

    // 서비스 2: 협력인지 (Cooperative Awareness, port 1237, PEDESTRIAN)
    //   인터페이스: Uu/wwan0 (5G gNB 경유, 광역 커버리지 우선)
    //   주파수 대역: 2.1 GHz, n1 (LTE/5G 광역 커버리지)
    double freqCoopAwareness   = 2.1e9;   // [Hz] 3GPP n1
    double bwCoopAwareness     = 20e6;    // [Hz] 20 MHz

    // 서비스 3: 협력주행 (Cooperative Driving, ports 1235/1236, PC5)
    //   인터페이스: PC5/v2v0 (SL 직접링크, gNB 불필요)
    //   주파수 대역: 5.89 GHz, n47 (ITS-G5 전용, ETSI EN 302 663)
    double freqCoopDriving     = 5.89e9;  // [Hz] 3GPP n47 ITS — SIRIUS 지원 목록에 포함, 변경 없음
    // SIRIUS 하드웨어가 지원하는 대역폭은 {10, 20, 40} MHz뿐이므로 40MHz로 설정.
    double bwCoopDriving       = 40e6;    // [Hz] 40 MHz

    // BWP 매핑:
    //   BWP0 = Uu 대역 A (원격제어,  3.5 GHz n78)
    //   BWP1 = Uu 대역 B (협력인지,  2.1 GHz n1)   ← 새로 만듦
    //   BWP2 = SL PC5    (협력주행,  5.89 GHz n47)  ← 기존 BWP1에서 이동
    double centralFrequencyBandUu   = freqRemoteControl;  // BWP0
    double bandwidthBandUu          = bwRemoteControl;
    double centralFrequencyBandUuCa = freqCoopAwareness;  // BWP1 (협력인지)
    double bandwidthBandUuCa        = bwCoopAwareness;
    double centralFrequencyBandSl   = freqCoopDriving;    // BWP2 (협력주행)
    double bandwidthBandSl          = bwCoopDriving;

    uint32_t rbWidth                = 12 * 30000;
    uint16_t rbNum = static_cast<uint16_t>(bandwidthBandSl / rbWidth);

    std::string tddPattern = "DL|DL|DL|DL|DL|F|UL|UL|UL|UL|";
    // countUl=4, SL 400슬롯/sec → PC5 이론 최대 ~40 Mbps
    std::string slBitMap   = "1|1|1|1|1|1|1|1|1|1|1|1|";
    // 12비트: 12 % 4 = 0 ✓

    // 5G-NR Rel-16 Mode 2 (Distributed Scheduling) — TR 38.885 §5.4.4
    // μ=1 (30kHz SCS): 고속 주행(140 km/h) Doppler 허용, 0.5ms 슬롯
    uint16_t numerologyBwpSl      = 1;    // μ=1: 30kHz SCS
    // slSensingWindow: ms 단위 (슬롯 수 아님). 유효값: 100 또는 1100
    // Rel-16 TS 38.214 §8.1.2.1: T_sensing ≥ 500ms (μ=1 기준) → 1100ms 선택
    uint16_t slSensingWindow      = 1100; // 1100ms (μ=1에서 ≥500ms 요건 충족)
    // slSelectionWindow: 슬롯 단위. 유효값: 1, 5, 10, 20
    uint16_t slSelectionWindow    = 20;   // 20 슬롯 = 10ms at μ=1
    uint16_t slSubchannelSize     = 50;  // 10/20/25와 비교 측정 시 차이 미미(가용 PRB가 모두 ~100)
    // SIRIUS 스펙은 {1,2,4}, ns-3 구현은 {1,2,3}만 허용 -> 교집합 {1,2} 중 2 사용.
    // (기존 3은 SIRIUS 스펙 위반이었고, 4는 ns-3에서 불가)
    uint16_t slMaxNumPerReserve   = 2;
    // 유효값: 1, 2, 3 (ns-3 NR SL 구현 제한)
    double   slProbResourceKeep   = 0.0;
    uint16_t slMaxTxTransNumPssch = 2;  // 재전송 없이 스루풋 극대화
    // SIRIUS 스펙: sl-ResourceReservePeriodList는 value1=0이어야 하고 100을 포함해야 함.
    // ToD는 동적 그랜트(m_dynamic=true)를 쓰므로 RRI는 SPS 주기로 쓰이지 않는다.
    uint16_t reservationPeriod    = 100;
    // NOTE (PHASE C-2): 기본값 false 를 **바꾸지 않았다**. false 에서 mode-2 는
    // sensing 기반이 아니라 random resource selection 으로 동작한다. 그 결과
    // NrSlUeMac::ComputeCbr() 가 호출되지 않아 직접 CBR 측정이 나오지 않는다
    // (CbrMeasurement 트레이스는 연결되어 있으나 발화하지 않는다).
    //
    // 이것을 켜는 것은 자원 선택 방식 자체를 바꾸는 scenario semantics 변경이고
    // C-2 의 구현 범위가 아니다. CLI 로만 노출해 C-3 가 두 조건을 모두 평가한
    // 뒤 결정할 수 있게 한다. 기본값을 바꾸지 않았으므로 동작은 불변이다.
    bool     enableSensing        = false;
    uint16_t t1                   = 2;
    // 선택 윈도우 상한. 동적 그랜트는 LC당 미결 그랜트가 1개뿐이라 처리량이
    // 대략 1/((T1+T2)/2 슬롯)으로 제한된다. T2min(20슬롯, μ=1)이 하한이라 20이 최소.
    // 33 -> 20 으로 낮춰 그랜트 주기를 8.75ms -> 5.5ms 로 단축(실측 +64%).
    uint16_t t2                   = 20;
    // LC당 동시에 미결 상태로 둘 수 있는 동적 그랜트 수. 1이면 그랜트 왕복
    // (T1..T2 선택윈도우+피드백)마다 PDU 하나만 보낼 수 있어 처리량이 묶인다.
    // SL HARQ 프로세스 풀(16개)이 실질 상한.
    // 실측 掃引 결과 8에서 포화(16도 동일). 1 -> 8 로 2.69 -> 3.07 Mbps.
    uint16_t maxOutstandingGrants = 8;
    int      slThresPsschRsrp     = -128;
    // 3GPP TR 38.901 채널 랜덤성 활성화 (현실적 페이딩 시뮬레이션용)
    // true → 섀도잉 + 공간적 채널 변동 반영 (SINR 시계열 변동성 증가)
    bool     enableChannelRandomness = true; //수정 (2026.06.09)
    uint16_t channelUpdatePeriod  = 500;  // 100 → 500 (ms), DDQN 불필요한 HO 감소
    // 27(QAM256 최댓값, 고SINR 전제)에서 4로 하향 — 이 모듈(NrSlUeMacScheduler
    // 계열)에는 적응형 MCS 스케줄러가 없고 NrSlUeMacSchedulerFixedMcs(고정값)만
    // 존재하므로, 실측 pc5_rssi 분포(중앙값 -88.3dBm)에 맞춰 고정값 자체를
    // 낮춘다. Noise ≈ -174+10log10(40MHz)+NF(7dB) ≈ -91dBm 기준으로 중앙값
    // SINR ≈ 2~3dB → QPSK 저코딩률 대역(MCS 2~5)이 되어야 디코딩 가능.
    // MCS 4를 보수적 기본값으로 선택(더 나쁜 구간은 MCS와 무관하게 SINR 자체가
    // 마이너스라 HARQ 재전송(위 harqEnabled)에 의존할 수밖에 없음).
    // uint8_t 였을 때 ns-3 CommandLine이 값을 '문자'로 파싱해(--mcs=4 -> '4'=52)
    // 유효범위(0~28)를 벗어나는 버그가 있었다. uint16_t로 변경해 숫자로 파싱되게 한다.
    // TB 크기와 BLER의 트레이드오프 최적점(실측). MCS 4는 100PRB/9심볼에서 TB가
    // 약 609B라 1034B 패킷이 매번 분할되어 처리량이 반토막 났고, MCS 20 이상은
    // 실측 SINR(≈3dB)에서 BLER이 급증한다. MCS 10이 2.69Mbps로 최적.
    uint16_t mcs                  = 10;
    uint16_t uuPort               = 1234;   // ToD Uu 포트
    uint16_t slPort               = 1235;   // CP SL 포트

    // --- 링크 지연 ---
    double remoteDelayMs = 20.0;   // PGW↔RemoteHost 링크 지연 (ms)
    // RemoteHost↔RSU 백홀 지연 (ms). 기존 하드코딩 값 1.0을 그대로 기본값으로
    // 둔다(동작 불변). Uu 경로는 remoteDelay(20ms)를 거치므로 두 값이 다르면
    // RAT 비교에 백홀 구조 차이가 섞인다 — 공정 비교 시 remoteDelay와 맞출 것.
    double rsuBackhaulDelayMs = 1.0;

    // --- ToD ACK 인과 A/B 실험 ---
    bool disableTodAck = false; // true = per-packet Uu ACK 송신과 pending ACK 계측만 비활성화

    // --- 난수 시드 ---
    // 지금까지는 SetRun을 호출하지 않아 RngRun이 항상 기본값 1로 고정됐다.
    // 다중 시드 평가를 위해 CLI로 노출한다. 기본값 1 = 기존 동작과 동일.
    // uint64_t: RngSeedManager::SetRun() 의 시그니처와 일치시킨다. episode 별
    // rngRun 정책(master_seed*100000 + episode)이 9자리 값을 만들기 때문에
    // uint32_t 였다면 master_seed >= 42950 에서 조용히 wrap 했다.
    uint64_t rngRun = 1;

    // --- [추가] ToD 트래픽 ---
    double todDataRateMbps = 4.0;  // [RADIO-DIAG]

    bool generateInitialPosGnuScript = false;
    bool generateGifGnuScript        = false;
    std::string simTag    = "default";
    std::string outputDir = "./";
    bool        saveDb    = true;
    std::string mobilityFile  =
        "contrib/nr/examples/nr-v2x-examples/vehicle_mobility.tcl";
    // [GEOMETRY-PROBE] 정적 RSU-차량 거리 실험용. 0보다 크면 SUMO 이동성 대신
    // carNode를 RSU로부터 이 거리(수평, m)만큼 떨어진 고정 위치에 배치한다.
    double testDistanceM = -1.0;
    uint32_t numRsu        = 1;
    uint32_t numVehicles   = 1;
    int      numGnb        = 1;
    // Active surrounding vehicles: SL BWP only, ego 와 동일한 PC5 resource pool 에서
    // CAM 을 송신해 mode-2 contention 을 만든다. RL 제어 대상이 아니다.
    // PROVISIONAL candidate levels {0, 9, 18} — C-3 이전에 freeze 하지 않는다.
    uint32_t numActiveVeh  = 0;
    // 송신하는 차량 수. -1(기본)이면 numActiveVeh 와 같다(= C-2 동작 그대로).
    //
    // 왜 설치 수와 송신 수를 분리하는가: C-2.5 통제 실험에서 --camSilent 로
    // CAM 을 0개 보내도 numActiveVeh 를 바꾸면 ego 의 uu_arrivals 가 8173 ->
    // 8085 -> 7854 로 변했다. 즉 차이의 상당 부분이 부하가 아니라 **디바이스
    // 존재**(gNB 노드 ID 이동, 공유 ThreeGppChannelModel 의 link-pair 평가
    // 순서, 이벤트 삽입 순서)에서 온다. 이것은 stream 할당으로 제거되지
    // 않는다 — 모든 stream 을 고정해도 남는다.
    //
    // 따라서 load level 을 비교할 때는 **설치 population 을 고정**하고 송신하는
    // 대수만 바꾼다. 그러면 구조가 동일하고 유일한 차이가 트래픽이 된다.
    int32_t  numCamTx      = -1;
    // numSumoNodes 는 numActiveVeh 파싱 후 유도한다 (아래 CommandLine 파싱 뒤).

    // ── DDQN 실행 모드 / 결정 주기 / 최소 유지시간 ──────────────
    std::string ratMode              = "ddqn"; // "ddqn" | "dup0".."dup100"
    // TODO: 10ms는 실측 근거 없이 잠정 설정된 값. Ettifos SDK 실제 metric
    // 갱신주기 확인 후 재검토 필요.
    double      dqnDecisionIntervalMs = 100.0; // DDQN 결정 주기 (ms)
    bool        enablePc5Debug        = false; // PHY/scheduler/link-budget 계측 on/off
    // 출력 정책. 빈 문자열이면 기존 동작(소스 디렉토리에 기록) — 과거 스크립트와
    // 아카이브 reader 가 그대로 동작한다. 값이 주어지면 그 디렉토리 아래에만
    // 기록한다. episode 마다 rngRun 이 바뀌는 새 시나리오에서는 반드시 지정해야
    // 수백 개 CSV 가 소스 트리에 쌓이지 않는다.
    std::string scenarioOutDir       = "";
    // 기본값 유지 = 기존 순차 실행과 완전 하위 호환.
    std::string shmSegment           = "TvtAiSeg";
    // LEGACY_ONLY: MP-QUIC 시대 handover 계측(Scenario1/2/3 CSV, D1/D2,
    // tvt_ho_delay_*.csv, tvt_ho_times_*.csv) 재현용. 기본 OFF.
    bool        enableLegacyHoTelemetry = false;
    // VALIDATION CONTROL: active vehicle 을 설치하되 CAM 을 보내지 않는다.
    bool        camSilent               = false;
    // 초기 duplication level. 경로 선택이 아니라 중복 정도다.
    std::string startDup              = "dup0";

    // ── RSU 위치 ────────────────────────────────────────────
    // 과제 개념도(Uu 커버리지 안에 RSU PC5 커버리지가 동심원으로 들어가고 CAV가
    // 그 안을 통과)에 맞춰 gNB(5599.8, 6597.4) 바로 아래로 옮겼다.
    //
    // 이전 좌표 (4980, 6600)은 차량 출발점(5104.99, 6694.51) 바로 옆이라
    // PC5 유효 구간이 t=1~27s와 388~415s로 양 끝에만 생겼고, 그 결과 핸드오버
    // 9회가 전부 초반/후반에 몰리고 28~388s의 360초 동안 0회였다.
    //
    // vehicle_mobility.tcl을 분석해 검증한 값 (415s 기준):
    //   최소거리 53.5m @ t=299s   (이전 55.4m @ t=18s와 동등 → PC5 최대 품질 유지)
    //   180m 이내 167s (40.2%)    구간 49~121s, 273~366s   (이전 55s / 13.3%)
    //   100m 이내  23s ( 5.5%)    구간 105~115s, 293~304s
    // 180m를 기준으로 잡은 근거: 기존 eval에서 에이전트가 PC5를 선택한 494스텝의
    // RSU 거리가 중앙값 113m, 95%tile 174m, 최대 184m였다.
    //
    // gNB와 수평 0m로 겹치지만 대역(3.5GHz n78 vs 5.89GHz n47)·경로손실 모델
    // (ThreeGppUma vs ThreeGppV2vHighway)·spectrum channel이 모두 분리되어 있어
    // 상호 간섭은 없다. 높이도 25m/10m로 다르다.
    double rsuX      = 5599.8;
    double rsuY      = 6597.4;
    double rsuHeight = 10.0;

    // ── gNB 위치 ────────────────────────────────────────────
    // 기본값 = 도로 중심에서 남쪽 300 m (TVT_CONTENDED_CORRIDOR_V1 candidate).
    // 기존 동작을 재현하려면 --gnbY=6597.4 로 RSU 와 같은 수평 좌표에 둔다.
    double gnbX      = 5599.8;
    double gnbY      = 6297.4;
    double gnbZ      = 25.0;

    // ── Mobility window ─────────────────────────────────────
    // trace 의 어느 구간을 episode 로 쓸지. 0 이면 trace 처음부터.
    //
    // TVT_CONTENDED_CORRIDOR_V2 = 289.0 -> window t=289..317.
    //   d_RSU 118 -> 54 -> 123 m, 최근접 episode t=9.9 s
    //   approach 9 s / departure 18 s, corr(d_RSU,d_gNB) = -0.256
    //
    // 왜 v1 의 92.0 에서 옮겼는가 (PHASE C-4): offset 92 에서는 최근접점이
    // episode t=18.20 s 로 SAFETY 요구 구간 [15,25) 안에 있었다. closest
    // region(d_RSU<=100 m) 11 초 중 8 초(73%)가 SAFETY 와 겹쳐, "SAFETY 요구가
    // 활성"과 "PC5 기하가 가장 좋음"이 결정론적으로 정렬됐다. 289 에서는 그
    // 겹침이 0% 가 되고 최근접점이 VIDEO-only 구간으로 들어간다.
    //
    // 선택은 trace 기하만으로 했고 ns-3 QoS 결과를 보기 전에 동결했다.
    double mobilityStartOffsetS = 289.0;
    std::string loadTransitionsSpec = "";

    // ----------------------------------------------------------
    // CommandLine 파싱
    // ----------------------------------------------------------
    CommandLine cmd(__FILE__);
    cmd.AddValue("logging",             "Enable logging",                       logging);
    cmd.AddValue("speed",               "Vehicle speed (m/s)",                  speed);
    cmd.AddValue("useIPv6",             "Use IPv6",                             useIPv6);
    cmd.AddValue("packetSizeBe",        "BE packet size (bytes)",               udpPacketSizeBe);
    cmd.AddValue("dataRateBe",          "BE data rate (kbps)",                  dataRateBe);
    cmd.AddValue("simTime",             "Simulation time (s)",                  simTime);
    cmd.AddValue("slBearerActivationTime", "SL bearer activation time (s)",     slBearersActivationTime);
    cmd.AddValue("freqRemoteControl",   "원격제어  Uu BWP0 center freq (Hz)",  freqRemoteControl);
    cmd.AddValue("bwRemoteControl",     "원격제어  Uu BWP0 bandwidth (Hz)",     bwRemoteControl);
    cmd.AddValue("freqCoopAwareness",   "협력인지  Uu BWP1 center freq (Hz)",  freqCoopAwareness);
    cmd.AddValue("bwCoopAwareness",     "협력인지  Uu BWP1 bandwidth (Hz)",     bwCoopAwareness);
    cmd.AddValue("freqCoopDriving",     "협력주행  SL BWP2 center freq (Hz)",  freqCoopDriving);
    cmd.AddValue("bwCoopDriving",       "협력주행  SL BWP2 bandwidth (Hz)",     bwCoopDriving);
    cmd.AddValue("tddPattern",          "TDD pattern string",                   tddPattern);
    cmd.AddValue("slBitMap",            "SL bitmap string",                     slBitMap);
    cmd.AddValue("numerologyBwpSl",     "SL BWP numerology",                    numerologyBwpSl);
    cmd.AddValue("slSensingWindow",     "SL sensing window (ms)",               slSensingWindow);
    cmd.AddValue("enableSensing",
                 "NR SL mode-2 sensing-based resource selection. false(기본)=random "
                 "selection, 이 경우 직접 CBR 측정이 발생하지 않는다. "
                 "PHASE C-2 는 기본값을 바꾸지 않았다 — C-3 결정 사항",
                 enableSensing);
    cmd.AddValue("slSelectionWindow",   "SL selection window",                  slSelectionWindow);
    cmd.AddValue("slSubchannelSize",    "SL subchannel size (RBs)",             slSubchannelSize);
    cmd.AddValue("slMaxNumPerReserve",  "SL max reserved resources",            slMaxNumPerReserve);
    cmd.AddValue("slProbResourceKeep",  "SL resource keep probability",         slProbResourceKeep);
    cmd.AddValue("slMaxTxTransNumPssch","SL max PSSCH retransmissions",         slMaxTxTransNumPssch);
    cmd.AddValue("enableSensing",       "Enable SL sensing",                    enableSensing);
    cmd.AddValue("t1",                  "SL selection window start (slots)",    t1);
    cmd.AddValue("t2",                  "SL selection window end (slots)",      t2);
    cmd.AddValue("maxOutstandingGrants","LC당 동시 미결 동적 그랜트 수 (1~16)",  maxOutstandingGrants);
    cmd.AddValue("slThresPsschRsrp",    "SL PSSCH RSRP threshold (dBm)",        slThresPsschRsrp);
    cmd.AddValue("enableChannelRandomness", "Enable shadowing",                 enableChannelRandomness);
    cmd.AddValue("channelUpdatePeriod", "Channel update period (ms)",           channelUpdatePeriod);
    cmd.AddValue("mcs",                 "SL MCS",                               mcs);
    cmd.AddValue("remoteDelay",         "PGW↔RemoteHost link delay (ms)",       remoteDelayMs);
    cmd.AddValue("rsuBackhaulDelay",    "RemoteHost↔RSU 백홀 지연 (ms)",        rsuBackhaulDelayMs);
    cmd.AddValue("rngRun",              "ns-3 RngRun (독립 시드 번호, 기본 1)", rngRun);
    cmd.AddValue("disableTodAck",       "per-packet ToD Uu ACK 송신/계측 비활성화", disableTodAck);
    cmd.AddValue("todDataRate",         "ToD downlink data rate (Mbps)",        todDataRateMbps);
    cmd.AddValue("outputDir",           "Output directory",                     outputDir);
    cmd.AddValue("simTag",              "Simulation tag",                       simTag);
    cmd.AddValue("saveDb",              "Save SQLite database",                 saveDb);
    cmd.AddValue("generateInitialPosGnuScript", "Generate init pos gnuplot",    generateInitialPosGnuScript);
    cmd.AddValue("generateGifGnuScript",        "Generate mobility GIF script", generateGifGnuScript);
    cmd.AddValue("mobilityFile",                "SUMO mobility trace 파일 경로", mobilityFile);
    cmd.AddValue("testDistanceM",       "[GEOMETRY-PROBE] 정적 RSU-차량 거리(m), >0이면 SUMO 대신 고정 배치", testDistanceM);
    cmd.AddValue("mobilityStartOffsetS",
                 "mobility trace 시작 오프셋(s). 0 이면 기존 동작. "
                 "기본 92.0 = TVT_CONTENDED_CORRIDOR_V1 window",
                 mobilityStartOffsetS);
    cmd.AddValue("loadTransitionsSec",
                 "Comma-separated nominal CAM load transition times (s). "
                 "Empty keeps the frozen 28 s list 8,15,21,25.",
                 loadTransitionsSpec);
    cmd.AddValue("gnbX", "gNB x (m)", gnbX);
    cmd.AddValue("gnbY", "gNB y (m). 기본 6297.4 = 도로 남쪽 300m. "
                         "6597.4 면 RSU 와 동일 수평 좌표(기존 동작)", gnbY);
    cmd.AddValue("gnbZ", "gNB height (m)", gnbZ);
    cmd.AddValue("numCamTx",
                 "CAM 을 송신하는 active vehicle 수 (<=numActiveVeh). -1 이면 "
                 "numActiveVeh 전부. load level 비교 시 numActiveVeh 를 고정하고 "
                 "이 값만 바꾸면 구조가 동일한 통제 비교가 된다",
                 numCamTx);
    cmd.AddValue("numActiveVeh",
                 "Active surrounding vehicles (SL only, CAM transmitters sharing the "
                 "ego's PC5 pool). PROVISIONAL candidates 0/9/18",
                 numActiveVeh);
    cmd.AddValue("ratMode",             "실행 모드: ddqn(기본) | dup0/dup25/dup50/dup75/dup100 (고정 duplication level)", ratMode);
    cmd.AddValue("dqnDecisionIntervalMs", "DDQN 결정 주기 (ms)",                 dqnDecisionIntervalMs);
    cmd.AddValue("enablePc5Debug",      "PC5 PHY/scheduler/link-budget 계측 출력", enablePc5Debug);
    cmd.AddValue("shmSegment",
                 "ns3-ai 공유 메모리 세그먼트 이름. 동시 실행 격리용 "
                 "INFRASTRUCTURE 전용 — 과학적 수량을 바꾸지 않는다. Python 측 "
                 "--shm-segment 와 반드시 동일한 값을 주어야 한다. 기본 "
                 "\"TvtAiSeg\" 는 기존 동작을 그대로 재현한다.",
                 shmSegment);
    cmd.AddValue("scenarioOutDir",
                 "CSV 출력 디렉토리. 비우면 소스 디렉토리(기존 동작). "
                 "episode 별 rngRun 을 쓰는 시나리오에서는 지정 필수",
                 scenarioOutDir);
    cmd.AddValue("camSilent",
                 "VALIDATION ONLY: active vehicle 을 설치하되 CAM 을 송신하지 않는다. "
                 "'차량 존재' 와 '차량 송신' 의 효과를 분리하는 통제용. 기본 false",
                 camSilent);
    cmd.AddValue("enableLegacyHoTelemetry",
                 "LEGACY: MP-QUIC 시대 handover 계측(Scenario1/2/3, D1/D2, ho CSV) 재현. "
                 "물리 handover 는 존재하지 않는다 — 진단 출력 전용, 기본 off",
                 enableLegacyHoTelemetry);
    cmd.AddValue("rsuX",                "RSU X 좌표 (m)",                       rsuX);
    cmd.AddValue("rsuY",                "RSU Y 좌표 (m)",                       rsuY);
    cmd.AddValue("rsuHeight",           "RSU 높이 (m)",                         rsuHeight);
    cmd.AddValue("startDup",            "초기 duplication level: dup0(기본)/dup25/dup50/dup75/dup100", startDup);
    cmd.Parse(argc, argv);

    // RngRun을 명시적으로 고정한다. 호출하지 않으면 항상 1이 쓰여 모든 런이
    // 동일 궤적을 반복한다(결정론적 재실행일 뿐 통계적 독립 표본이 아니다).
    NS_ABORT_MSG_IF(rngRun == 0, "rngRun must be >= 1");
    RngSeedManager::SetRun(rngRun);

    // 노드 수는 역할별 구성에서 유도한다:
    //   [0 .. numRsu-1]                       RSU
    //   [numRsu]                              ego (Uu + SL)
    //   [numRsu+1 .. numRsu+numActiveVeh]     active surrounding (SL only)
    // background(mobility-only) 차량은 두지 않는다 — 3GPP V2V 경로손실 모델의
    // LOS/NLOS/NLOSv 판정이 확률적이라 좌표만 있는 노드는 blockage 를 만들지
    // 못하고 시뮬레이션 시간만 소비한다.
    const uint32_t numSumoNodes = numRsu + numVehicles + numActiveVeh;
    NS_ABORT_MSG_IF(numCamTx > static_cast<int32_t>(numActiveVeh),
                    "numCamTx (" << numCamTx << ") exceeds numActiveVeh ("
                    << numActiveVeh << ")");
    const uint32_t camTxCount = (numCamTx < 0)
                                ? numActiveVeh
                                : static_cast<uint32_t>(numCamTx);
    g_disableTodAck = disableTodAck;
    // 경로 간 재조립(reordering)은 존재하지 않으며 존재해서도 안 된다:
    // duplication 은 의도된 중복 전송이고, 수신측은 logical seq 로 first-arrival
    // 을 취하고 뒤늦은 복제본을 억제한다. 은퇴한 MP-QUIC 재정렬 버퍼 CLI
    // (reorderHoldMs/reorderBypass)는 제거했다.
    std::cout << ">>> [TOD-ACK] "
              << (g_disableTodAck ? "DISABLED (treatment)" : "ENABLED (baseline)")
              << std::endl;
    std::cout << ">>> [SEED] RngSeed=" << RngSeedManager::GetSeed()
              << " RngRun=" << RngSeedManager::GetRun() << std::endl;
    std::cout << ">>> [BACKHAUL] remoteDelay=" << remoteDelayMs << "ms"
              << " rsuBackhaulDelay=" << rsuBackhaulDelayMs << "ms"
              << (std::abs(remoteDelayMs - rsuBackhaulDelayMs) > 1e-9
                      ? "  [비대칭 — RAT 비교에 백홀 구조 차이가 포함됨]"
                      : "  [대칭]")
              << std::endl;

    // duplication level 이름 <-> action index. 한 곳에만 둔다.
    auto dupLevelFromName = [](const std::string& name) -> int {
        if (name == "dup0")   return TVT_AI_ACTION_DUP_0;
        if (name == "dup25")  return TVT_AI_ACTION_DUP_25;
        if (name == "dup50")  return TVT_AI_ACTION_DUP_50;
        if (name == "dup75")  return TVT_AI_ACTION_DUP_75;
        if (name == "dup100") return TVT_AI_ACTION_DUP_100;
        return -1;
    };
    // 은퇴한 경로 선택 모드를 조용히 무시하지 않는다: 의미가 사라졌으므로
    // 하드 에러로 거부한다.
    if (ratMode == "uu" || ratMode == "pc5" || ratMode == "both")
    {
        NS_ABORT_MSG("ratMode '" << ratMode << "' is RETIRED with contract v6. "
                     "The action is a duplication degree, not a path: use "
                     "dup0/dup25/dup50/dup75/dup100.");
    }
    if (ratMode != "ddqn" && dupLevelFromName(ratMode) < 0)
    {
        NS_ABORT_MSG("unknown ratMode '" << ratMode
                     << "' — expected ddqn or dup0/dup25/dup50/dup75/dup100");
    }
    g_ratMode              = ratMode;
    g_forcedDuplicationLevel = (ratMode == "ddqn") ? -1 : dupLevelFromName(ratMode);
    NS_ABORT_MSG_IF(shmSegment.empty(), "shmSegment must not be empty");
    g_shmSegment            = shmSegment;
    g_dqnDecisionIntervalMs = dqnDecisionIntervalMs;
    g_enablePc5Debug        = enablePc5Debug;
    if (!scenarioOutDir.empty())
    {
        g_scenOutDir = scenarioOutDir;
        if (g_scenOutDir.back() != '/') g_scenOutDir += '/';
        // 디렉토리는 호출자가 만들어 둔다. 없으면 ofstream 이 조용히 실패하므로
        // 여기서 바로 확인하고 거부한다 — 결과가 조용히 사라지면 안 된다.
        std::ofstream probe(g_scenOutDir + ".tvt_outdir_probe");
        NS_ABORT_MSG_IF(!probe.is_open(),
                        "scenarioOutDir is not writable: " << g_scenOutDir);
        probe.close();
        std::remove((g_scenOutDir + ".tvt_outdir_probe").c_str());
    }
    std::cout << ">>> [OUTDIR] " << g_scenOutDir << std::endl;
    g_enableLegacyHoTelemetry = enableLegacyHoTelemetry;
    g_camSilent               = camSilent;
    NS_ABORT_MSG_IF(g_dqnDecisionIntervalMs < 100.0,
                    "dqnDecisionIntervalMs must be >= 100 ms because RAT application "
                    "is finalized by the 100 ms action-switch monitor; shorter intervals "
                    "would overwrite a pending decision before its applied result.");

    if (dupLevelFromName(startDup) < 0)
    {
        NS_ABORT_MSG("unknown startDup '" << startDup
                     << "' — expected dup0/dup25/dup50/dup75/dup100");
    }
    g_currentAction = dupLevelFromName(startDup);
    std::cout << ">>> [TX-SEMANTICS] primary=Uu(path " << TVT_AI_PRIMARY_PATH
              << ") secondary=PC5(path " << TVT_AI_SECONDARY_PATH
              << ")  policy=fixed_uu_primary_v1"
              << "  start=" << startDup << std::endl;

    NS_ABORT_IF(freqRemoteControl  > 6e9);
    NS_ABORT_IF(freqCoopAwareness  > 6e9);
    NS_ABORT_IF(freqCoopDriving    > 6e9);
    // 파생 변수 갱신 (cmd.AddValue로 재정의된 경우)
    centralFrequencyBandUu   = freqRemoteControl;
    bandwidthBandUu          = bwRemoteControl;
    centralFrequencyBandUuCa = freqCoopAwareness;
    bandwidthBandUuCa        = bwCoopAwareness;
    centralFrequencyBandSl   = freqCoopDriving;
    bandwidthBandSl          = bwCoopDriving;

    // 핸드오버 파라미터를 전역 변수에 반영
    g_simEndTime        = simTime.GetSeconds();
    if (!loadTransitionsSpec.empty())
    {
        std::vector<double> parsed;
        std::stringstream ss(loadTransitionsSpec);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty()) parsed.push_back(std::stod(tok));
        NS_ABORT_MSG_IF(parsed.empty(),
                        "loadTransitionsSec given but parsed to nothing: "
                        << loadTransitionsSpec);
        for (size_t i = 1; i < parsed.size(); ++i)
            NS_ABORT_MSG_IF(parsed[i] <= parsed[i - 1],
                            "loadTransitionsSec must be strictly increasing");
        NS_ABORT_MSG_IF(parsed.front() < 0.0,
                        "loadTransitionsSec must not start before 0");
        g_loadNominalSec = parsed;
    }
    g_todDataRateMbps   = todDataRateMbps;
    // g_ue1NodeId는 노드 생성 후 설정 (아래에서 설정)

    if (logging) {
        LogLevel logLevel = (LogLevel)(LOG_PREFIX_TIME | LOG_PREFIX_NODE | LOG_LEVEL_ALL);
        LogComponentEnable("OnOffApplication", logLevel);
        LogComponentEnable("PacketSink", logLevel);
    }

    // ----------------------------------------------------------
    // 시뮬레이션 기본값
    // ----------------------------------------------------------
    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(999999999));
    Config::SetDefault("ns3::NrGnbMac::NumRbPerRbg",     UintegerValue(1));

    // ----------------------------------------------------------
    // UE(차량) 노드 생성 + SUMO 이동성 설치
    // UE0 = RSU (고정), UE1~N = SUMO 차량 (이동)
    // ----------------------------------------------------------
    NodeContainer allSlUesContainer;
    allSlUesContainer.Create(numSumoNodes);

    // ── RSU/차량 컨테이너 분리 ──────────────────────────────
    //   egoVehicle:        인덱스 numRsu (기존 carNode, Uu+SL)
    //   activeVehicles:    인덱스 numRsu+1 ~ numRsu+numActiveVeh (SL만)
    //   backgroundVehicles: 나머지 (MobilityModel만, NR 스택 없음)
    NodeContainer rsuOnlyContainer;
    for (uint32_t i = 0; i < numRsu; ++i)
        rsuOnlyContainer.Add(allSlUesContainer.Get(i));

    NodeContainer activeVehContainer;
    for (uint32_t i = numRsu + 1; i <= numRsu + numActiveVeh; ++i)
        activeVehContainer.Add(allSlUesContainer.Get(i));

    // Ns2MobilityHelper 대상은 EGO 하나뿐이다. vehicle_mobility.tcl 은 node_(0)
    // 하나만 정의하므로, active vehicle 을 이 범위에 넣으면 트레이스 항목이
    // 없어 (0,0,0) 에 머문다. 따라서 ego 만 넘기고 active 는 아래에서 별도의
    // 합성 이동성을 받는다.
    NodeContainer vehOnlyContainer;
    vehOnlyContainer.Add(allSlUesContainer.Get(numRsu));

    // ── RSU: ConstantPositionMobilityModel 설치 (고정 좌표) ─────
    // RSU는 SUMO 트레이스(mobilityFile) 대상이 아니므로 독립적으로 고정 배치.
    MobilityHelper rsuMobHelper;
    rsuMobHelper.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    rsuMobHelper.Install(rsuOnlyContainer);
    if (rsuOnlyContainer.GetN() > 0) {
        rsuOnlyContainer.Get(0)->GetObject<MobilityModel>()
            ->SetPosition(Vector(rsuX, rsuY, rsuHeight));
    }

    // ── 차량 전체: SUMO 이동성 적용 (Ns2MobilityHelper) ────────
    // Install()(무인자)은 트레이스의 node_(i)를 전역 ns3::NodeList의 nodeId i와
    // 매칭한다. allSlUesContainer가 RSU(인덱스 0..numRsu-1) 다음에 차량을
    // 생성하므로, 전역 매칭을 쓰면 node_(0)이 RSU(전역 nodeId 0)에 적용되어
    // carNode가 트레이스를 못 받는다. 대신 Install(begin,end)는 트레이스의
    // node_(i)를 넘겨준 컨테이너의 i번째 원소에 매칭하므로, 차량만 담은
    // vehOnlyContainer를 넘겨 node_(0) → vehOnlyContainer의 첫 차량(carNode)이
    // 되도록 한다. RSU는 위에서 이미 별도로 ConstantPositionMobilityModel +
    // 고정 좌표를 설치했으므로 트레이스 대상에서 제외된다.
    // [GEOMETRY-PROBE] testDistanceM > 0이면 SUMO 이동성 대신 RSU로부터 고정 거리에
    // 정적 배치(V2V_Highway 거리 의존성 실험용). 그 외에는 기존 SUMO 트레이스 그대로.
    if (testDistanceM > 0.0)
    {
        MobilityHelper staticCarMob;
        staticCarMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        staticCarMob.Install(vehOnlyContainer);
        vehOnlyContainer.Get(0)->GetObject<MobilityModel>()
            ->SetPosition(Vector(rsuX + testDistanceM, rsuY, 1.5));
        std::cout << "[GEOMETRY-PROBE] carNode 정적 배치: RSU로부터 수평거리 " << testDistanceM
                  << "m (차량 높이 1.5m)" << std::endl;
    }
    else
    {
        const std::string windowTrace =
            BuildOffsetMobilityTrace(mobilityFile, mobilityStartOffsetS, g_scenOutDir);
        Ns2MobilityHelper ns2mobility(windowTrace);
        ns2mobility.Install(vehOnlyContainer.Begin(), vehOnlyContainer.End());
    }
    if (!g_carNodePtr) g_carNodePtr = allSlUesContainer.Get(numRsu);

    // ── Active surrounding vehicles: 합성 이동성 ──────────────────────────
    // 목적은 ego 를 막거나 인위적 충돌을 만드는 것이 아니라 PC5 pool 을 공유하는
    // 공간적으로 그럴듯한 송신자 분포를 만드는 것이다. ego corridor 를 따라
    // 양방향 차선에 배치하고 등속으로 움직인다.
    //
    // 모든 난수는 ns-3 전역 RNG stream 에서 뽑으므로 rngRun 에 의해 결정된다:
    //   같은 master seed + 같은 episode -> 동일 배치
    //   다른 episode                    -> 다른 배치
    if (numActiveVeh > 0)
    {
        MobilityHelper activeMob;
        activeMob.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
        activeMob.Install(activeVehContainer);

        // ego 궤적의 x 범위(trace 분석값 5027..6172) 안에 균등 배치하고, 도로
        // 중심선(rsuY) 기준으로 차선 오프셋을 준다. 속도는 ego trace 의 실측
        // 범위(평균 7.2, 최대 14.7 m/s) 안에서 뽑는다.
        // DOMAIN_B. 생성 시점에 Stream 을 지정해 전역 카운터를 소비하지 않는다.
        Ptr<UniformRandomVariable> uX =
            CreateObjectWithAttributes<UniformRandomVariable>(
                "Stream", IntegerValue(kStreamVehX),
                "Min", DoubleValue(kActiveVehXMin),
                "Max", DoubleValue(kActiveVehXMax));
        Ptr<UniformRandomVariable> uSpd =
            CreateObjectWithAttributes<UniformRandomVariable>(
                "Stream", IntegerValue(kStreamVehSpeed),
                "Min", DoubleValue(kActiveVehSpeedMin),
                "Max", DoubleValue(kActiveVehSpeedMax));
        Ptr<UniformRandomVariable> uJit =
            CreateObjectWithAttributes<UniformRandomVariable>(
                "Stream", IntegerValue(kStreamVehLane),
                "Min", DoubleValue(-kActiveVehLaneJitterM),
                "Max", DoubleValue(kActiveVehLaneJitterM));

        // 전체 conceptual population 을 먼저 뽑고 앞에서부터 쓴다. numActiveVeh
        // 와 무관하게 draw 수가 고정되므로 L1 의 앞 9대와 L2 의 앞 9대가
        // 비트 단위로 같다 — L1 -> L2 차이는 "기존 9대가 달라지는 것"이 아니라
        // "뒤에 9대가 추가되는 것"이 된다.
        std::vector<double> popX(kActiveVehMaxPopulation);
        std::vector<double> popSpd(kActiveVehMaxPopulation);
        std::vector<double> popLaneJit(kActiveVehMaxPopulation);
        for (uint32_t i = 0; i < kActiveVehMaxPopulation; ++i)
        {
            popX[i]       = uX->GetValue();
            popSpd[i]     = uSpd->GetValue();
            popLaneJit[i] = uJit->GetValue();
        }

        for (uint32_t i = 0; i < activeVehContainer.GetN(); ++i)
        {
            // 차선: 짝수 index 는 동쪽(+x), 홀수는 서쪽(-x). 2차선 양방향.
            const bool   eastbound = (i % 2 == 0);
            const double laneY     = rsuY
                                   + (eastbound ? kActiveVehLaneOffsetM
                                                : -kActiveVehLaneOffsetM)
                                   + popLaneJit[i];
            Ptr<MobilityModel> mm = activeVehContainer.Get(i)->GetObject<MobilityModel>();
            mm->SetPosition(Vector(popX[i], laneY, kActiveVehHeightM));
            activeVehContainer.Get(i)->GetObject<ConstantVelocityMobilityModel>()
                ->SetVelocity(Vector(eastbound ? popSpd[i] : -popSpd[i], 0.0, 0.0));
        }
        // [GEOMETRY-PROBE] prefix 일관성 검증용: 앞 N대의 생성값을 그대로 찍는다.
        std::cout << ">>> [RNG-DOMAIN] policy=" << kRngControlPolicy
                  << "  population=" << kActiveVehMaxPopulation
                  << "  used=" << activeVehContainer.GetN() << std::endl;
        for (uint32_t i = 0; i < activeVehContainer.GetN(); ++i)
        {
            Ptr<MobilityModel> mm = activeVehContainer.Get(i)->GetObject<MobilityModel>();
            const Vector pos = mm->GetPosition();
            const Vector vel = mm->GetVelocity();
            std::cout << ">>> [VEH-INIT] i=" << i
                      << " x=" << std::fixed << std::setprecision(6) << pos.x
                      << " y=" << pos.y
                      << " vx=" << vel.x << std::endl;
        }
        std::cout << ">>> [ACTIVE-VEH] installed=" << activeVehContainer.GetN()
                  << "  x=[" << kActiveVehXMin << "," << kActiveVehXMax << "]"
                  << "  laneOffset=" << kActiveVehLaneOffsetM << "m"
                  << "  speed=[" << kActiveVehSpeedMin << "," << kActiveVehSpeedMax
                  << "]m/s" << std::endl;
    }

    // carNode = 첫 번째 차량 (인덱스 numRsu)
    g_ue1NodeId       = allSlUesContainer.Get(numRsu)->GetId();
    g_carNodePtr      = allSlUesContainer.Get(numRsu);
    g_allUesContainer = allSlUesContainer;

    // RSU 전용 NodeContainer (인덱스 0..numRsu-1)
    for (uint32_t i = 0; i < numRsu; ++i)
        g_rsuNodesGlobal.Add(allSlUesContainer.Get(i));

    // ----------------------------------------------------------
    // [추가] gNB 노드 생성 + 위치 설정 (고정 좌표, ETRI 테스트베드 배치 기준)
    // ----------------------------------------------------------
    NodeContainer gNbNodes;
    gNbNodes.Create(numGnb);

    MobilityHelper gnbMobility;
    gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMobility.Install(gNbNodes);

    if (gNbNodes.GetN() > 0) {
        // gNB 좌표는 CLI 파라미터다 (--gnbX/--gnbY/--gnbZ).
        //
        // 기본값은 도로 중심(5599.8, 6597.4)에서 남쪽 300 m 다. RSU 와 수평
        // 좌표가 같던 기존 배치에서는 두 경로의 대규모 기하가 완전히 겹쳐
        // corr(d_RSU, d_gNB) = +1.000 이었다 — Uu 와 PC5 가 함께 좋아지고 함께
        // 나빠져 redundancy 가 의미를 갖기 어려웠다. 남쪽 300 m 로 옮기면
        // window t=92..120 에서 corr = +0.098 이 되고 d_gNB 는 217..326 m 로
        // 현실적인 macro cell 거리를 유지한다. RSU 는 지역 hotspot 으로 남는다.
        //
        // 더 멀리 밀면 상관은 더 떨어지지만 Uu 가 비현실적으로 약해지고
        // "Uu 는 늘 나쁘다"는 또 다른 상수 편향이 생긴다. 완전 독립이 목표가
        // 아니라 identical failure mode 의 제거가 목표다.
        gNbNodes.Get(0)->GetObject<MobilityModel>()
            ->SetPosition(Vector(gnbX, gnbY, gnbZ));
        std::cout << ">>> [GEOMETRY] gNB=(" << gnbX << "," << gnbY << "," << gnbZ << ")"
                  << "  RSU=(" << rsuX << "," << rsuY << "," << rsuHeight << ")"
                  << "  horizontal_separation="
                  << std::sqrt((gnbX - rsuX) * (gnbX - rsuX)
                             + (gnbY - rsuY) * (gnbY - rsuY)) << "m"
                  << std::endl;
    }

    // ----------------------------------------------------------
    // EPC 헬퍼 및 NR 헬퍼 생성
    // ----------------------------------------------------------
    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetEpcHelper(epcHelper);

    Ptr<NrSlHelper> nrSlHelper = CreateObject<NrSlHelper>();
    nrSlHelper->SetEpcHelper(epcHelper);

    // SidelinkInfo: 차량↔RSU/서버 ToD 릴레이용 (Groupcast, dstL2Id=255)
    uint32_t dstL2Id = 255;
    SidelinkInfo slInfo;
    slInfo.m_castType    = SidelinkInfo::CastType::Groupcast;
    slInfo.m_dstL2Id     = dstL2Id;
    slInfo.m_rri         = MilliSeconds(reservationPeriod);
    slInfo.m_dynamic     = true;
    slInfo.m_pdb         = delayBudget;
    slInfo.m_harqEnabled = harqEnabled;

    // SidelinkInfo: 차량간 BSM용 (Groupcast, dstL2Id=254)
    SidelinkInfo slInfoBsm;
    slInfoBsm.m_castType    = SidelinkInfo::CastType::Groupcast;
    slInfoBsm.m_dstL2Id     = 254;
    slInfoBsm.m_rri         = MilliSeconds(reservationPeriod);
    slInfoBsm.m_dynamic     = true;
    slInfoBsm.m_pdb         = delayBudget;
    slInfoBsm.m_harqEnabled = false;

    // ----------------------------------------------------------
    // 채널 모델 설정 — 3GPP TR 38.901 기반
    //   Uu (BWP0,BWP1): UMa (Urban Macro) — TR 38.901 Table 7.4.1-1
    //     ThreeGppUmaPropagationLossModel + NLOS ChannelConditionModel
    //     gNB 높이 25m, UE 1.5m, 도심 거시 셀 환경
    //   SL PC5 (BWP2): V2V-Highway — TR 38.901 Table 7.4.1-1
    //     ThreeGppV2vHighwayPropagationLossModel
    //     차량 높이 1.5m, 고속도로 LOS/NLOS 확률 모델
    //     μ=1(30kHz SCS) → 최대 도플러 fd = v/λ ≈ 140km/h@5.89GHz = 766Hz
    // ----------------------------------------------------------
    if (enableChannelRandomness) {
        Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod",
                           TimeValue(MilliSeconds(channelUpdatePeriod)));
        nrHelper->SetChannelConditionModelAttribute("UpdatePeriod",
                                                    TimeValue(MilliSeconds(channelUpdatePeriod)));
        nrHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true));
    } else {
        Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod",
                           TimeValue(MilliSeconds(0)));
        nrHelper->SetChannelConditionModelAttribute("UpdatePeriod",
                                                    TimeValue(MilliSeconds(0)));
        nrHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
    }


    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;


    CcBwpCreator::SimpleOperationBandConf bandConfUu(
        centralFrequencyBandUu, bandwidthBandUu, numCcPerBand,
        BandwidthPartInfo::UMi_StreetCanyon);
    bandConfUu.m_numCc = 1;
    OperationBandInfo bandUu = ccBwpCreator.CreateOperationBandContiguousCc(bandConfUu);
    nrHelper->InitializeOperationBand(&bandUu);


    CcBwpCreator::SimpleOperationBandConf bandConfUuCa(
        centralFrequencyBandUuCa, bandwidthBandUuCa, numCcPerBand,
        BandwidthPartInfo::UMi_StreetCanyon);
    OperationBandInfo bandUuCa = ccBwpCreator.CreateOperationBandContiguousCc(bandConfUuCa);
    nrHelper->InitializeOperationBand(&bandUuCa);


    // [수정] 근본 원인 ①: 위 설계 주석은 SL PC5를 V2V-Highway로 명시하지만
    // 실제로는 Uu와 동일한 UMi_StreetCanyon(비대칭 BS-UE 지오메트리용 셀룰러
    // 모델)이 적용되고 있었다 — PC5 SINR이 비현실적으로 크게 나온 핵심 원인
    // 중 하나. UMi_StreetCanyon → ThreeGppUmiStreetCanyonPropagationLossModel,
    // V2V_Highway → ThreeGppV2vHighwayPropagationLossModel로 서로 다른 경로손실
    // 모델(및 LOS/NLOS 채널 조건 모델)에 매핑되므로(nr-helper.cc InitUmi/
    // InitV2VHighway), 동일 차량 높이(1.5m)의 대칭 V2V 링크에 맞는 모델로
    // 교체한다.
    CcBwpCreator::SimpleOperationBandConf bandConfSl(
        centralFrequencyBandSl, bandwidthBandSl, numCcPerBand,
        BandwidthPartInfo::V2V_Highway);
    OperationBandInfo bandSl = ccBwpCreator.CreateOperationBandContiguousCc(bandConfSl);
    nrHelper->InitializeOperationBand(&bandSl);

    // BWP 벡터 구성
    // gNB: Uu BWP 1개만 (UuCA 포함 시 동시 DL로 LteInterference 중복 StartRx 크래시)
    BandwidthPartInfoPtrVector gNbBwps  = CcBwpCreator::GetAllBwps({bandUu});
    // UE: SL + Uu (BWP0=협력주행 SL, BWP1=원격제어 Uu)
    BandwidthPartInfoPtrVector allBwps  = CcBwpCreator::GetAllBwps({bandSl, bandUu});
    // SL BWP ID = 0 (allBwps에서 첫 번째)
    uint8_t bwpIdForSl  = 0;

    // ----------------------------------------------------------
    // [추가] PHY 전력 설정 (안테나 속성은 기본값 사용)
    // 주의: SetGnbAntennaAttribute / SetUeAntennaAttribute 사용 시
    //       BeamManager m_beamManager 초기화 타이밍 문제로 NS_ASSERT 크래시 발생.
    //       attack.cc 와 동일하게 기본 안테나 설정 유지.
    // ----------------------------------------------------------
    // gNB TxPower는 Uu 다운링크(gNB→UE) 한 종류뿐이므로 install 시점에 바로 설정.
    // UE 쪽 TxPower는 링크(Uu/PC5)·노드(carNode/RSU/activeVeh)별로 값이 다르므로
    // 아래 각 InstallUeDevice() 직전에 개별적으로 설정한다(블랭킷 설정 제거).
    nrHelper->SetGnbPhyAttribute("TxPower", DoubleValue(UU_GNB_TXPOWER_DBM));
    // [수정] NoiseFigure 속성이 어디서도 설정되지 않아 NrPhy 기본값인 0.0dB
    // (이상적인 무잡음 수신기, 비현실적)가 그대로 적용되고 있었다. 실제
    // 수신기는 항상 양의 잡음지수를 가지므로(전형적 UE 수신기 ~7dB) 현실적인
    // 값으로 명시한다. (Uu/PC5 RSSI는 이제 NrInterference/NrSlInterference의
    // RssiPerProcessedChunk 실측값을 직접 쓰므로 이 속성에 의존하지 않지만,
    // NrSlInterference의 SINR 계산 자체와 Uu 링크 예산에는 여전히 영향을 준다.)
    nrHelper->SetGnbPhyAttribute("NoiseFigure", DoubleValue(7.0));
    nrHelper->SetUePhyAttribute("NoiseFigure",  DoubleValue(7.0));

    // ----------------------------------------------------------
    // MAC / BWP 매니저 설정
    // NrSlUeMac: Uu + SL 모두 처리 (NrUeMac 상속)
    // ----------------------------------------------------------
    nrHelper->SetUeMacTypeId(NrSlUeMac::GetTypeId());
    nrHelper->SetUeMacAttribute("EnableSensing",    BooleanValue(enableSensing));
    nrHelper->SetUeMacAttribute("T1",               UintegerValue(static_cast<uint8_t>(t1)));
    nrHelper->SetUeMacAttribute("T2",               UintegerValue(t2));
    nrHelper->SetUeMacAttribute("ActivePoolId",     UintegerValue(0));
    nrHelper->SetUeMacAttribute("SlThresPsschRsrp", IntegerValue(slThresPsschRsrp));

    // BWP 매니저: QCI → BWP 라우팅
    //   carNode: GBR_MC_PUSH_TO_TALK → BWP1(SL, 5.89GHz)
    //   RSU/activeVeh(slOnlyBwps): GBR_MC_PUSH_TO_TALK → BWP0(SL, 5.89GHz)
    nrHelper->SetBwpManagerTypeId(TypeId::LookupByName("ns3::NrSlBwpManagerUe"));

    std::set<uint8_t> bwpIdContainer;
    // carSl(slOnlyBwps)은 BWP0=SL만 존재
    bwpIdContainer.insert(bwpIdForSl);  // BWP0: 협력주행 SL = 0

    // ----------------------------------------------------------
    // InternetStack 설치 (UE)
    // ----------------------------------------------------------
    InternetStackHelper internet;
    internet.Install(allSlUesContainer);

    // ----------------------------------------------------------
    // [추가] gNB 디바이스 설치 (Uu BWP만)
    // ----------------------------------------------------------
    NetDeviceContainer gNbDevices = nrHelper->InstallGnbDevice(gNbNodes, gNbBwps);

    // ----------------------------------------------------------
    // UE 디바이스 설치
    //   RSU: SL BWP(bandSl)만 (gNB attach 불필요, Uu PHY 오버헤드 제거)
    //   carNode: Uu + SL 전체 BWP (핸드오버 대상)
    // ----------------------------------------------------------
    NodeContainer carOnlyContainer;
    carOnlyContainer.Add(allSlUesContainer.Get(numRsu));

    // RSU/carSl/activeVeh: SL BWP만 설치 → Uu 채널 간섭 방지
    BandwidthPartInfoPtrVector slOnlyBwps = CcBwpCreator::GetAllBwps({bandSl});
    BandwidthPartInfoPtrVector uuOnlyBwps = CcBwpCreator::GetAllBwps({bandUu});

    // [수정] RSSI가 비정상적으로 크게 나오는 문제는 txPower를 낮춰 결과를
    // 우회 조정하는 대신 SlPsschRxCallback()의 계산 로직 자체를 고치는
    // 방향으로 대응했다(아래 참고). TxPower 자체는 링크/노드 역할별 4개
    // 상수(UU_UE/UU_GNB/PC5_UE/PC5_RSU_TXPOWER_DBM)로 InstallUeDevice() 직전에
    // 개별 설정한다.

    // RSU: slOnlyBwps → SL at BWP index 0 (PC5 다운링크, RSU)
    nrHelper->SetUeBwpManagerAlgorithmAttribute("GBR_MC_PUSH_TO_TALK", UintegerValue(0));
    nrHelper->SetUePhyAttribute("TxPower", DoubleValue(PC5_RSU_TXPOWER_DBM));
    NetDeviceContainer rsuSlNetDevContainer =
        nrHelper->InstallUeDevice(rsuOnlyContainer, slOnlyBwps);

    // carNode SL device: slOnlyBwps → SL at BWP index 0 (PC5 업링크, UE)
    nrHelper->SetUeBwpManagerAlgorithmAttribute("GBR_MC_PUSH_TO_TALK", UintegerValue(0));
    nrHelper->SetUePhyAttribute("TxPower", DoubleValue(PC5_UE_TXPOWER_DBM));
    NetDeviceContainer carSlNetDevContainer =
        nrHelper->InstallUeDevice(carOnlyContainer, slOnlyBwps);

    // carNode Uu device: uuOnlyBwps (gNB 연결용, Uu 업링크 UE)
    nrHelper->SetUePhyAttribute("TxPower", DoubleValue(UU_UE_TXPOWER_DBM));
    NetDeviceContainer carUuNetDevContainer =
        nrHelper->InstallUeDevice(carOnlyContainer, uuOnlyBwps);

    // activeVehicles: SL BWP만 설치 (BSM 브로드캐스트용) → SL at BWP index 0 (PC5 업링크, UE)
    nrHelper->SetUeBwpManagerAlgorithmAttribute("GBR_MC_PUSH_TO_TALK", UintegerValue(0));
    nrHelper->SetUePhyAttribute("TxPower", DoubleValue(PC5_UE_TXPOWER_DBM));
    NetDeviceContainer activeVehNetDevContainer =
        nrHelper->InstallUeDevice(activeVehContainer, slOnlyBwps);

    // ── DOMAIN_A: radio/channel stream 고정 ────────────────────────────────
    // 각 디바이스 그룹에 고정 base 를 준다. 이렇게 하면 active vehicle 을 몇 대
    // 설치하든 gNB / RSU / ego 의 채널·fading stream 번호가 바뀌지 않는다.
    // 이것이 없으면 numActiveVeh 가 자동 할당 카운터를 밀어 ego 의 Uu
    // realization 까지 바꾼다 (PHASE C-2 에서 관측된 uu_arrivals 변화).
    {
        const int64_t usedGnb    = nrHelper->AssignStreams(gNbDevices,            kStreamBaseGnb);
        const int64_t usedRsu    = nrHelper->AssignStreams(rsuSlNetDevContainer,  kStreamBaseRsuSl);
        const int64_t usedEgoSl  = nrHelper->AssignStreams(carSlNetDevContainer,  kStreamBaseEgoSl);
        const int64_t usedEgoUu  = nrHelper->AssignStreams(carUuNetDevContainer,  kStreamBaseEgoUu);
        const int64_t usedActive = nrHelper->AssignStreams(activeVehNetDevContainer,
                                                           kStreamBaseActiveVeh);
        std::cout << ">>> [RNG-DOMAIN] radio stream bases"
                  << "  gnb=" << kStreamBaseGnb       << "(+" << usedGnb    << ")"
                  << "  rsu=" << kStreamBaseRsuSl     << "(+" << usedRsu    << ")"
                  << "  ego_sl=" << kStreamBaseEgoSl  << "(+" << usedEgoSl  << ")"
                  << "  ego_uu=" << kStreamBaseEgoUu  << "(+" << usedEgoUu  << ")"
                  << "  active=" << kStreamBaseActiveVeh << "(+" << usedActive << ")"
                  << std::endl;
        // 그룹 간 침범이 생기면 조용히 상관된 realization 을 쓰게 되므로 거부한다.
        NS_ABORT_MSG_IF(kStreamBaseGnb   + usedGnb   > kStreamBaseRsuSl ||
                        kStreamBaseRsuSl + usedRsu   > kStreamBaseEgoSl ||
                        kStreamBaseEgoSl + usedEgoSl > kStreamBaseEgoUu ||
                        kStreamBaseEgoUu + usedEgoUu > kStreamBaseActiveVeh,
                        "radio RNG stream ranges overlap -- widen the bases");
    }

    NetDeviceContainer allSlUesNetDeviceContainer;
    allSlUesNetDeviceContainer.Add(rsuSlNetDevContainer);
    allSlUesNetDeviceContainer.Add(carSlNetDevContainer);
    allSlUesNetDeviceContainer.Add(carUuNetDevContainer);
    allSlUesNetDeviceContainer.Add(activeVehNetDevContainer);
    // backgroundVehicles: NR 스택 없음 (MobilityModel만)

    // RSU Rx Trace 연결
    Ptr<NetDevice> rsuDev = allSlUesNetDeviceContainer.Get(0);
    rsuDev->TraceConnectWithoutContext("PhyRxEnd", MakeCallback(&RxTraceCallback));

    // ----------------------------------------------------------
    // IP 주소 할당
    // ----------------------------------------------------------
    Ipv4InterfaceContainer ueIpIface =
        epcHelper->AssignUeIpv4Address(allSlUesNetDeviceContainer);

    // RSU IP 포워딩 활성화 (백홀 ↔ SL 릴레이 경로)
    for (uint32_t i = 0; i < numRsu; ++i) {
        Ptr<Ipv4> rsuIpv4 = allSlUesContainer.Get(i)->GetObject<Ipv4>();
        rsuIpv4->SetAttribute("IpForward", BooleanValue(true));
        for (uint32_t j = 0; j < rsuIpv4->GetNInterfaces(); ++j)
            rsuIpv4->SetForwarding(j, true);
    }

    // ----------------------------------------------------------
    // UE UpdateConfig
    // ----------------------------------------------------------
    for (auto it = allSlUesNetDeviceContainer.Begin();
         it != allSlUesNetDeviceContainer.End(); ++it) {
        DynamicCast<NrUeNetDevice>(*it)->UpdateConfig();
    }
    for (auto it = gNbDevices.Begin(); it != gNbDevices.End(); ++it) {
        DynamicCast<NrGnbNetDevice>(*it)->UpdateConfig();
    }


    // ----------------------------------------------------------
    // EPC 인프라: PGW 획득
    // ----------------------------------------------------------
    Ptr<Node> pgw = epcHelper->GetPgwNode();

    // ----------------------------------------------------------
    // Remote Host 노드 생성 (PGW에 직접 연결)
    // 구성: RemoteHost --(remoteDelay)-- PGW -- gNB -- UE
    // ----------------------------------------------------------
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);

    InternetStackHelper internetRemote;
    internetRemote.Install(remoteHostContainer);

    // PGW ↔ Remote Host 링크
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate",  DataRateValue(DataRate("100Gb/s")));
    // MilliSeconds()는 uint64_t만 받아 소수점 ms가 잘린다 — Seconds()로 정확히 준다.
    p2ph.SetChannelAttribute("Delay",    TimeValue(Seconds(remoteDelayMs / 1000.0)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    Ipv4Address remoteHostAddr [[maybe_unused]] = internetIpIfaces.GetAddress(1);

    // remoteHost ↔ RSU 백홀 (numRsu개, 루프)
    std::vector<NetDeviceContainer> bhDevices(numRsu);
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        Ipv4RoutingHelper::GetRouting<Ipv4StaticRouting>(
            remoteHost->GetObject<Ipv4>()->GetRoutingProtocol());

    for (uint32_t i = 0; i < numRsu; ++i) {
        PointToPointHelper bh;
        bh.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
        bh.SetChannelAttribute("Delay",   TimeValue(Seconds(rsuBackhaulDelayMs / 1000.0)));
        bhDevices[i] = bh.Install(remoteHost, allSlUesContainer.Get(i));

        Ipv4AddressHelper ipBh;
        std::string subnet = "10.0." + std::to_string(i) + ".0";
        ipBh.SetBase(subnet.c_str(), "255.255.255.0");
        Ipv4InterfaceContainer bhIface = ipBh.Assign(bhDevices[i]);
        g_rsuBackhaulAddrs.push_back(bhIface.GetAddress(1));

        std::string net = "10.0." + std::to_string(i) + ".0";
        remoteHostStaticRouting->AddNetworkRouteTo(
            Ipv4Address(net.c_str()), Ipv4Mask("255.255.255.0"),
            bhDevices[i].Get(0)->GetIfIndex());
    }

    // ----------------------------------------------------------
    // [추가] 모든 UE(RSU + 차량)를 gNB에 attach
    // RSU(UE0)는 실제 Uu 트래픽 없으나 Uu BWP(BWP0)가 설치되어 있음.
    // AttachToClosestEnb 없이 StartSlot이 time=0에 실행되면 m_spectrumPhy 등
    // 미초기화 포인터 접근으로 NS_ASSERT("m_ptr") 크래시 발생.
    // → 모든 UE를 attach하여 Uu PHY를 올바르게 초기화.
    // ----------------------------------------------------------
    nrHelper->AttachToClosestEnb(carUuNetDevContainer, gNbDevices);

    NetDeviceContainer carDevices;
    for (uint32_t i = 1; i < allSlUesNetDeviceContainer.GetN(); ++i) {
        carDevices.Add(allSlUesNetDeviceContainer.Get(i));
    }

    // ----------------------------------------------------------
    // SL 설정 (기존 유지, BWP ID만 1로 변경)
    // ----------------------------------------------------------
    std::string errorModel = "ns3::NrEesmIrT1";
    nrSlHelper->SetSlErrorModel(errorModel);
    nrSlHelper->SetUeSlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    nrSlHelper->SetNrSlSchedulerTypeId(NrSlUeMacSchedulerFixedMcs::GetTypeId());
    nrSlHelper->SetUeSlSchedulerAttribute("Mcs", UintegerValue(mcs));
    nrSlHelper->SetUeSlSchedulerAttribute("MaxOutstandingDynamicGrants",
                                          UintegerValue(maxOutstandingGrants));

    // carSl: slOnlyBwps(BWP0=SL)
    nrSlHelper->PrepareUeForSidelink(carSlNetDevContainer, bwpIdContainer);

    // RSU: slOnlyBwps(BWP0=SL) → {0}만 준비
    std::set<uint8_t> slOnlyBwpIdContainer;
    slOnlyBwpIdContainer.insert(0);
    nrSlHelper->PrepareUeForSidelink(rsuSlNetDevContainer, slOnlyBwpIdContainer);
    // activeVeh: slOnlyBwps(BWP0=SL만) → {0}만 준비 (BWP1 없으므로 GetMac(1) 크래시 방지)
    std::set<uint8_t> activeVehBwpContainer;
    activeVehBwpContainer.insert(0);
    nrSlHelper->PrepareUeForSidelink(activeVehNetDevContainer, activeVehBwpContainer);
    NetDeviceContainer slOnlyDevContainer;
    slOnlyDevContainer.Add(rsuSlNetDevContainer);
    slOnlyDevContainer.Add(activeVehNetDevContainer);

    // SL 리소스 풀 설정
    LteRrcSap::SlResourcePoolNr slResourcePoolNr;
    Ptr<NrSlCommResourcePoolFactory> ptrFactory = Create<NrSlCommResourcePoolFactory>();

    std::vector<std::bitset<1>> slBitMapVector;
    GetSlBitmapFromString(slBitMap, slBitMapVector);
    NS_ABORT_MSG_IF(slBitMapVector.empty(), "GetSlBitmapFromString failed");

    ptrFactory->SetSlTimeResources(slBitMapVector);
    ptrFactory->SetSlSensingWindow(slSensingWindow);
    ptrFactory->SetSlSelectionWindow(slSelectionWindow);
    ptrFactory->SetSlFreqResourcePscch(10);
    ptrFactory->SetSlSubchannelSize(slSubchannelSize);
    ptrFactory->SetSlMaxNumPerReserve(slMaxNumPerReserve);
    std::list<uint16_t> resourceReservePeriodList = {0, reservationPeriod};  // SIRIUS: value1=0 필수
    ptrFactory->SetSlResourceReservePeriodList(resourceReservePeriodList);
    // ── PSFCH 설정 (5G-NR Rel-16 HARQ 피드백 채널) ──────────────
    // harqEnabled=true(위)와 정합을 맞춘다 — NrSlUeMacSchedulerFixedMcs는
    // grant.harqEnabled = allocationInfo.m_harqEnabled && GetMac()->GetPsfchPeriod()
    // 로 계산하므로, PSFCH period가 0이면 harqEnabled=true를 설정해도 실제로는
    // HARQ가 꺼진 채로 동작한다(이전 상태의 실제 원인). ns-3 CreatePool()은
    // {0,1,2,4}만 허용하는데, SIRIUS 하드웨어 스펙상 confirmed 값은 {0,1}뿐이고
    // {2,4}는 TBD(미확인)이므로, 확인된 값 중 HARQ를 켜는 1을 사용한다.
    // psfchPeriod=1: 매 슬롯마다 PSFCH 전송 (HARQ ACK/NACK → 재전송 결정)
    // minTimeGapPsfch=3: PSSCH 마지막 슬롯과 PSFCH 사이 최소 3슬롯 보장 (3GPP TS 38.214,
    // 유효값은 {2,3}뿐)
    ptrFactory->SetSlPsfchPeriod(1);
    ptrFactory->SetSlMinTimeGapPsfch(3);
    LteRrcSap::SlResourcePoolNr pool = ptrFactory->CreatePool();
    slResourcePoolNr = pool;

    LteRrcSap::SlResourcePoolConfigNr slresoPoolConfigNr;
    slresoPoolConfigNr.haveSlResourcePoolConfigNr = true;
    uint16_t poolId = 0;
    LteRrcSap::SlResourcePoolIdNr slResourcePoolIdNr;
    slResourcePoolIdNr.id = poolId;
    slresoPoolConfigNr.slResourcePoolId = slResourcePoolIdNr;
    slresoPoolConfigNr.slResourcePool   = slResourcePoolNr;

    LteRrcSap::SlBwpPoolConfigCommonNr slBwpPoolConfigCommonNr;
    slBwpPoolConfigCommonNr.slTxPoolSelectedNormal[slResourcePoolIdNr.id] = slresoPoolConfigNr;

    LteRrcSap::Bwp bwp;
    bwp.numerology    = numerologyBwpSl;
    bwp.symbolsPerSlots = 14;
    bwp.bandwidth = static_cast<uint16_t>(bandwidthBandSl / 100000);  // 100kHz 단위
    bwp.rbPerRbg  = 1;

    LteRrcSap::SlBwpGeneric slBwpGeneric;
    slBwpGeneric.bwp             = bwp;
    slBwpGeneric.slLengthSymbols = LteRrcSap::GetSlLengthSymbolsEnum(14);
    slBwpGeneric.slStartSymbol   = LteRrcSap::GetSlStartSymbolEnum(0);

    LteRrcSap::SlBwpConfigCommonNr slBwpConfigCommonNr;
    slBwpConfigCommonNr.haveSlBwpGeneric            = true;
    slBwpConfigCommonNr.slBwpGeneric                = slBwpGeneric;
    slBwpConfigCommonNr.haveSlBwpPoolConfigCommonNr = true;
    slBwpConfigCommonNr.slBwpPoolConfigCommonNr     = slBwpPoolConfigCommonNr;

    // ── Dummy SL pool for Uu BWPs (BWP0, BWP1) ────────────────────────────
    // NrSlUeMac은 모든 BWP에 설치되므로 DoSlotIndication에서 m_slTxPool을
    // 무조건 역참조한다. m_slTxPool이 nullptr이면 ptr.h:718에서 크래시.
    // → Uu BWP용 더미 풀: SL 비트맵 전체를 0으로 설정해
    //   IsSidelinkSlot()이 항상 false → SL 스케줄링 없음(UL/SL 충돌 방지),
    //   그러나 m_slTxPool은 non-null → 크래시 방지.
    // Uu BWP numerology는 gNB PHY에서 조회.
    uint16_t gnbNumerology0 =
        DynamicCast<NrGnbNetDevice>(gNbDevices.Get(0))->GetPhy(0)->GetNumerology();

    std::string uuSlBitMap = "0|0|0|0|0|0|0|0|0|0|0|0|";
    // uuSlBitMap 12개: Uu BWP (μ=0) countUl=4, 12%4=0 ✓
    std::vector<std::bitset<1>> uuSlBitMapVector;
    GetSlBitmapFromString(uuSlBitMap, uuSlBitMapVector);

    Ptr<NrSlCommResourcePoolFactory> uuPoolFactory = Create<NrSlCommResourcePoolFactory>();
    uuPoolFactory->SetSlTimeResources(uuSlBitMapVector);
    uuPoolFactory->SetSlSensingWindow(slSensingWindow);
    uuPoolFactory->SetSlSelectionWindow(slSelectionWindow);
    uuPoolFactory->SetSlFreqResourcePscch(10);
    uuPoolFactory->SetSlSubchannelSize(slSubchannelSize);
    uuPoolFactory->SetSlMaxNumPerReserve(slMaxNumPerReserve);
    std::list<uint16_t> uuReservePeriodList = {reservationPeriod};  // 0 제거: RRI=0 미지원
    uuPoolFactory->SetSlResourceReservePeriodList(uuReservePeriodList);
    uuPoolFactory->SetSlPsfchPeriod(0);  // 4 → 0 (실제 SL pool과 통일)
    uuPoolFactory->SetSlMinTimeGapPsfch(3);
    LteRrcSap::SlResourcePoolNr uuPoolNr = uuPoolFactory->CreatePool();

    LteRrcSap::SlResourcePoolConfigNr uuPoolConfigNr;
    uuPoolConfigNr.haveSlResourcePoolConfigNr = true;
    LteRrcSap::SlResourcePoolIdNr uuPoolIdNr;
    uuPoolIdNr.id = 0;
    uuPoolConfigNr.slResourcePoolId = uuPoolIdNr;
    uuPoolConfigNr.slResourcePool   = uuPoolNr;

    LteRrcSap::SlBwpPoolConfigCommonNr uuBwpPoolConfigCommonNr;
    uuBwpPoolConfigCommonNr.slTxPoolSelectedNormal[uuPoolIdNr.id] = uuPoolConfigNr;

    // BWP0 더미 설정 (원격제어 Uu: 3.5 GHz)
    LteRrcSap::Bwp bwpUu0;
    bwpUu0.numerology      = gnbNumerology0;
    bwpUu0.symbolsPerSlots = 14;
    bwpUu0.bandwidth       = static_cast<uint16_t>(bandwidthBandUu / 100000);
    bwpUu0.rbPerRbg        = 1;

    LteRrcSap::SlBwpGeneric slBwpGenericUu0;
    slBwpGenericUu0.bwp             = bwpUu0;
    slBwpGenericUu0.slLengthSymbols = LteRrcSap::GetSlLengthSymbolsEnum(14);
    slBwpGenericUu0.slStartSymbol   = LteRrcSap::GetSlStartSymbolEnum(0);

    LteRrcSap::SlBwpConfigCommonNr slBwpConfigUu0;
    slBwpConfigUu0.haveSlBwpGeneric            = true;
    slBwpConfigUu0.slBwpGeneric                = slBwpGenericUu0;
    slBwpConfigUu0.haveSlBwpPoolConfigCommonNr = true;
    slBwpConfigUu0.slBwpPoolConfigCommonNr     = uuBwpPoolConfigCommonNr;

    // BWP1 더미 설정 (협력인지 Uu: 2.1 GHz)
    LteRrcSap::Bwp bwpUu1;
    bwpUu1.numerology      = gnbNumerology0;
    bwpUu1.symbolsPerSlots = 14;
    bwpUu1.bandwidth       = static_cast<uint16_t>(bandwidthBandUuCa / 100000);
    bwpUu1.rbPerRbg        = 1;

    LteRrcSap::SlBwpGeneric slBwpGenericUu1;
    slBwpGenericUu1.bwp             = bwpUu1;
    slBwpGenericUu1.slLengthSymbols = LteRrcSap::GetSlLengthSymbolsEnum(14);
    slBwpGenericUu1.slStartSymbol   = LteRrcSap::GetSlStartSymbolEnum(0);

    LteRrcSap::SlBwpConfigCommonNr slBwpConfigUu1;
    slBwpConfigUu1.haveSlBwpGeneric            = true;
    slBwpConfigUu1.slBwpGeneric                = slBwpGenericUu1;
    slBwpConfigUu1.haveSlBwpPoolConfigCommonNr = true;
    slBwpConfigUu1.slBwpPoolConfigCommonNr     = uuBwpPoolConfigCommonNr;

    LteRrcSap::SlFreqConfigCommonNr slFreConfigCommonNr;
    slFreConfigCommonNr.slBwpList[0] = slBwpConfigCommonNr;  // SL at BWP0

    LteRrcSap::TddUlDlConfigCommon tddUlDlConfigCommon;
    tddUlDlConfigCommon.tddPattern = tddPattern;

    LteRrcSap::SlPreconfigGeneralNr slPreconfigGeneralNr;
    slPreconfigGeneralNr.slTddConfig = tddUlDlConfigCommon;

    LteRrcSap::SlUeSelectedConfig slUeSelectedPreConfig;
    NS_ABORT_MSG_UNLESS(slProbResourceKeep <= 1.0, "slProbResourceKeep must be 0~1");
    slUeSelectedPreConfig.slProbResourceKeep = slProbResourceKeep;

    LteRrcSap::SlPsschTxParameters psschParams;
    psschParams.slMaxTxTransNumPssch = static_cast<uint8_t>(slMaxTxTransNumPssch);
    LteRrcSap::SlPsschTxConfigList pscchTxConfigList;
    pscchTxConfigList.slPsschTxParameters[0] = psschParams;
    slUeSelectedPreConfig.slPsschTxConfigList = pscchTxConfigList;

    LteRrcSap::SidelinkPreconfigNr slPreConfigNr;
    slPreConfigNr.slPreconfigGeneral            = slPreconfigGeneralNr;
    slPreConfigNr.slUeSelectedPreConfig         = slUeSelectedPreConfig;
    slPreConfigNr.slPreconfigFreqInfoList[0]    = slFreConfigCommonNr;

    nrSlHelper->InstallNrSlPreConfiguration(carSlNetDevContainer, slPreConfigNr);
    // RSU: slOnlyBwps → BWP0=SL
    LteRrcSap::SlFreqConfigCommonNr slFreConfigSlOnly;
    slFreConfigSlOnly.slBwpList[0] = slBwpConfigCommonNr;  // SL at BWP0
    LteRrcSap::SidelinkPreconfigNr slPreConfigSlOnly;
    slPreConfigSlOnly.slPreconfigGeneral         = slPreconfigGeneralNr;
    slPreConfigSlOnly.slUeSelectedPreConfig      = slUeSelectedPreConfig;
    slPreConfigSlOnly.slPreconfigFreqInfoList[0] = slFreConfigSlOnly;
    // RSU: BWP0=Uu dummy + BWP bwpIdForSl=SL
    nrSlHelper->InstallNrSlPreConfiguration(rsuSlNetDevContainer, slPreConfigSlOnly);
    // activeVeh: BWP0=SL만 (slOnlyBwps, BWP1 없음)
    LteRrcSap::SlFreqConfigCommonNr slFreConfigActiveVeh;
    slFreConfigActiveVeh.slBwpList[0] = slBwpConfigCommonNr;
    LteRrcSap::SidelinkPreconfigNr slPreConfigActiveVeh;
    slPreConfigActiveVeh.slPreconfigGeneral         = slPreconfigGeneralNr;
    slPreConfigActiveVeh.slUeSelectedPreConfig      = slUeSelectedPreConfig;
    slPreConfigActiveVeh.slPreconfigFreqInfoList[0] = slFreConfigActiveVeh;
    nrSlHelper->InstallNrSlPreConfiguration(activeVehNetDevContainer, slPreConfigActiveVeh);

    // carUuDevice용 더미 SL 설정 (NrSlUeMac 크래시 방지)
    {
        LteRrcSap::SlFreqConfigCommonNr slFreConfigUuOnly;
        slFreConfigUuOnly.slBwpList[0] = slBwpConfigUu0;  // dummy pool

        LteRrcSap::SidelinkPreconfigNr slPreConfigUuOnly;
        slPreConfigUuOnly.slPreconfigGeneral         = slPreconfigGeneralNr;
        slPreConfigUuOnly.slUeSelectedPreConfig      = slUeSelectedPreConfig;
        slPreConfigUuOnly.slPreconfigFreqInfoList[0] = slFreConfigUuOnly;

        std::set<uint8_t> uuOnlyBwpIdSet;
        uuOnlyBwpIdSet.insert(0);

        nrSlHelper->PrepareUeForSidelink(carUuNetDevContainer, uuOnlyBwpIdSet);
        nrSlHelper->InstallNrSlPreConfiguration(carUuNetDevContainer, slPreConfigUuOnly);
    }

    // ----------------------------------------------------------
    // [추가] 비연결 UE의 Uu BWP(BWP0, BWP1) numerology 수동 설정
    //
    // 문제: ConfigUeParams는 SL BWP(BWP2)만 SetNumerology를 호출함.
    //       Uu BWP(BWP0, BWP1)은 비연결 UE(RSU 등)에서 SCS=0 상태로 남음.
    //       시뮬레이션 시작 시 DoSetInitialBandwidth가 SCS=0으로 호출되면 abort.
    // 해결: gNB PHY의 numerology(=0, 15kHz SCS)를 모든 UE의 BWP0/BWP1에 복사.
    //       연결 UE는 AttachToClosestEnb에서 덮어쓰므로 영향 없음.
    // ----------------------------------------------------------
    {
        uint16_t gnbNumerology =
            DynamicCast<NrGnbNetDevice>(gNbDevices.Get(0))->GetPhy(0)->GetNumerology();
        for (auto it = allSlUesNetDeviceContainer.Begin();
             it != allSlUesNetDeviceContainer.End(); ++it)
        {
            Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(*it);
            if (ueDev)
            {
                // slOnlyBwps 디바이스(RSU/activeVeh)는 BWP 0만 존재
                for (uint32_t bIdx = 0; bIdx < ueDev->GetCcMapSize(); ++bIdx)
                {
                    ueDev->GetPhy(bIdx)->SetNumerology(gnbNumerology);
                }
            }
        }
    }

    // 랜덤 스트림 할당
    int64_t stream = 1;
    stream += nrHelper->AssignStreams(carSlNetDevContainer, stream);
    stream += nrHelper->AssignStreams(carUuNetDevContainer, stream);
    stream += nrHelper->AssignStreams(slOnlyDevContainer, stream);
    stream += nrSlHelper->AssignStreams(carSlNetDevContainer, stream);
    stream += nrSlHelper->AssignStreams(carUuNetDevContainer, stream);
    stream += nrSlHelper->AssignStreams(slOnlyDevContainer, stream);

    // ----------------------------------------------------------
    // TX/RX UE 분류 (기존 유지)
    // ----------------------------------------------------------
    NodeContainer txSlUes;
    NodeContainer rxSlUes;
    NetDeviceContainer txSlUesNetDevice;
    NetDeviceContainer rxSlUesNetDevice;

    txSlUes.Add(allSlUesContainer);
    rxSlUes.Add(allSlUesContainer);
    txSlUesNetDevice.Add(allSlUesNetDeviceContainer);
    rxSlUesNetDevice.Add(allSlUesNetDeviceContainer);

    // ----------------------------------------------------------
    // SL Bearer 활성화 — RSU(TX) / carNode(RX) 분리
    // BIDIRECTIONAL 사용 시 carNode도 SL TX 리소스를 예약하여
    // RSU PSSCH와 충돌 → NS_ASSERT 크래시. TX/RX 방향을 분리하여 방지.
    // ----------------------------------------------------------
    Ipv4Address groupAddress4("225.1.0.0");

    // RSU 디바이스 (0 ~ numRsu-1): SL TRANSMIT
    NetDeviceContainer rsuNetDevCont;
    for (uint32_t i = 0; i < numRsu; ++i)
        rsuNetDevCont.Add(allSlUesNetDeviceContainer.Get(i));

    // carNode 1대만 RECEIVE (ToD 수신 전용)
    NetDeviceContainer carNetDevCont;
    carNetDevCont.Add(allSlUesNetDeviceContainer.Get(numRsu));

    // slPort Bearer: RSU=TRANSMIT, carNode=RECEIVE
    Ptr<LteSlTft> tftBidirTx = Create<LteSlTft>(
        LteSlTft::Direction::TRANSMIT,
        groupAddress4, slPort, slInfo);
    nrSlHelper->ActivateNrSlBearer(slBearersActivationTime,
                                    rsuNetDevCont, tftBidirTx);

    Ptr<LteSlTft> tftBidirRx = Create<LteSlTft>(
        LteSlTft::Direction::RECEIVE,
        groupAddress4, slPort, slInfo);
    nrSlHelper->ActivateNrSlBearer(slBearersActivationTime,
                                    carNetDevCont, tftBidirRx);

    // uuPort Bearer: RSU=TRANSMIT, carNode=RECEIVE
    Ptr<LteSlTft> tftTodTx = Create<LteSlTft>(
        LteSlTft::Direction::TRANSMIT,
        groupAddress4, uuPort, slInfo);
    nrSlHelper->ActivateNrSlBearer(slBearersActivationTime,
                                    rsuNetDevCont, tftTodTx);

    Ptr<LteSlTft> tftTodRx = Create<LteSlTft>(
        LteSlTft::Direction::RECEIVE,
        groupAddress4, uuPort, slInfo);
    nrSlHelper->ActivateNrSlBearer(slBearersActivationTime,
                                    carNetDevCont, tftTodRx);

    // BSM은 단일 bearer(slPort=1235)로 전송 (Broadcast, dstL2Id=0xFFFF).
    // 패킷 크기로 타입 구분: 41B=BSM(GpsBroadcast)

    // 차량 BSM TRANSMIT Bearer (numRsu ~ numSumoNodes-1)
    /*
    {
        NetDeviceContainer vehBsmDevCont;
        for (uint32_t i = numRsu + 1; i < numSumoNodes; ++i)  // carNode 제외
            vehBsmDevCont.Add(allSlUesNetDeviceContainer.Get(i));

        Ptr<LteSlTft> tftVehBsmTx = Create<LteSlTft>(
            LteSlTft::Direction::TRANSMIT,
            groupAddress4, slPort, slInfoBsm);
        nrSlHelper->ActivateNrSlBearer(slBearersActivationTime,
                                        vehBsmDevCont, tftVehBsmTx);
    }
    */

    // activeVehicles BSM TRANSMIT Bearer (dstL2Id=254, SL only)
    {
        Ptr<LteSlTft> tftActiveBsmTx = Create<LteSlTft>(
            LteSlTft::Direction::TRANSMIT,
            groupAddress4, slPort, slInfoBsm);
        nrSlHelper->ActivateNrSlBearer(slBearersActivationTime,
                                        activeVehNetDevContainer, tftActiveBsmTx);
    }

    // ----------------------------------------------------------
    // RSU(UE0) SL 수신 소켓 설정
    // ----------------------------------------------------------
    Ptr<Socket> slRecvSocket = Socket::CreateSocket(
        allSlUesContainer.Get(0), UdpSocketFactory::GetTypeId());

    Ptr<NetDevice> rsuNetDev = allSlUesNetDeviceContainer.Get(0);
    Ptr<Ipv4>      rsuIpv4   = allSlUesContainer.Get(0)->GetObject<Ipv4>();
    uint32_t rsuIfIndex = rsuIpv4->GetInterfaceForDevice(rsuNetDev);

    slRecvSocket->BindToNetDevice(rsuNetDev);
    int bindResult = slRecvSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), slPort));

    Ipv4StaticRoutingHelper multicast;
    Ptr<Ipv4StaticRouting> rsuStaticRouting = multicast.GetStaticRouting(rsuIpv4);
    std::vector<uint32_t> outputIfs{rsuIfIndex};
    // source = UE1(Car)의 IP (ueIpIface.GetAddress(1) = 7.0.0.2 등)
    // RSU 자신 IP를 source로 쓰면 "RSU→RSU" 멀티캐스트만 매칭되어 UE1발 패킷 수신 불가
    rsuStaticRouting->AddMulticastRoute(
        ueIpIface.GetAddress(1),
        groupAddress4, rsuIfIndex, outputIfs);

    slRecvSocket->SetAllowBroadcast(true);

    if (bindResult == 0) {
        // BSM 수신 불필요
        // slRecvSocket->SetRecvCallback(MakeCallback(&NotifySlRxSizeDispatch));
    }

    // ----------------------------------------------------------
    // [트래픽 1] ego 차량 BSM — 현재 시나리오에서는 미설치.
    // g_slSendSocket 이 nullptr 로 남으므로 SendSlPacket() 은 즉시 반환한다.
    // PHASE C 에서 SAFETY 트래픽을 실제로 싣게 되면 여기서 소켓을 만든다.
    // ----------------------------------------------------------
    Ptr<Node> carNode = allSlUesContainer.Get(numRsu);  // 첫 번째 차량

    // ----------------------------------------------------------
    // [트래픽 2] active surrounding vehicle BSM 브로드캐스팅
    //   BSM 100ms 주기, groupAddress4, SL groupcast.
    //   numActiveVeh == 0 인 현재 시나리오에서는 0회 실행된다 — 즉 PC5
    //   경쟁 송신자가 존재하지 않는다. multi-vehicle 시나리오의 진입점이다.
    // ----------------------------------------------------------
    //
    // CAM 위상 무작위화: 모든 차량이 같은 순간에 송신하면 mode-2 충돌 패턴이
    // 인위적으로 동기화된다. 각 차량의 첫 송신 시각을 [0, kCamIntervalMs) 안에서
    // 뽑는다. UniformRandomVariable 은 ns-3 전역 RNG stream 을 쓰므로 rngRun 이
    // 위상을 결정한다 — 같은 master seed + episode 면 동일, 다른 episode 면 다르다.
    if (numActiveVeh > 0)
    {
        // DOMAIN_C. 고정 population 만큼 먼저 뽑아 prefix 를 보존한다.
        Ptr<UniformRandomVariable> uPhase =
            CreateObjectWithAttributes<UniformRandomVariable>(
                "Stream", IntegerValue(kStreamCamPhase),
                "Min", DoubleValue(0.0),
                "Max", DoubleValue(kCamIntervalMs));
        std::vector<double> popPhase(kActiveVehMaxPopulation);
        for (uint32_t i = 0; i < kActiveVehMaxPopulation; ++i)
            popPhase[i] = uPhase->GetValue();
        double phaseSum = 0.0, phaseMin = 1e18, phaseMax = -1e18;
        for (uint32_t i = 0; i < camTxCount; ++i) {
            Ptr<Node> vehNode = allSlUesContainer.Get(numRsu + 1 + i);
            Ptr<Socket> bsmSock = Socket::CreateSocket(
                vehNode, UdpSocketFactory::GetTypeId());
            bsmSock->BindToNetDevice(activeVehNetDevContainer.Get(i));
            bsmSock->SetAllowBroadcast(true);
            bsmSock->Connect(InetSocketAddress(groupAddress4, slPort));
            const double phaseMs = popPhase[i];
            std::cout << ">>> [CAM-PHASE] i=" << i << " phase_ms="
                      << std::fixed << std::setprecision(6) << phaseMs << std::endl;
            phaseSum += phaseMs;
            phaseMin = std::min(phaseMin, phaseMs);
            phaseMax = std::max(phaseMax, phaseMs);
            Simulator::Schedule(Seconds(kCamStartSec) + MilliSeconds(phaseMs),
                                &SendBsmForNode, vehNode, bsmSock);
        }
        std::cout << ">>> [CAM-TX] installed=" << numActiveVeh
                  << " transmitting=" << camTxCount
                  << "  payload=" << kCamPayloadBytes << "B"
                  << "  interval=" << kCamIntervalMs << "ms"
                  << "  start=" << kCamStartSec << "s"
                  << "  phase_ms[min/mean/max]=" << std::fixed << std::setprecision(1)
                  << phaseMin << "/" << (phaseSum / numActiveVeh) << "/" << phaseMax
                  << "  offered=" << (camTxCount * 1000.0 / kCamIntervalMs) << "pkt/s"
                  << std::endl;
    }
    // load schedule 은 numActiveVeh 와 무관하게 항상 계산한다. L0 에서는 켤
    // 송신자가 없으므로 toggle 을 스케줄하지 않지만, jitter 는 동일하게 뽑혀
    // 로그에 남는다 — 그래야 L0/L1/L2 의 전이 시각이 같다고 말할 수 있다.
    ScheduleCamLoadPhases(simTime.GetSeconds(), camTxCount > 0);
    if (camTxCount == 0) g_camEnabled = false;

    // ----------------------------------------------------------
    // [추가] 라우팅 설정
    // ----------------------------------------------------------
    // UE 기본 게이트웨이
    for (uint32_t u = 0; u < allSlUesContainer.GetN(); ++u) {
        Ptr<Node> uNode = allSlUesContainer.Get(u);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(uNode->GetObject<Ipv4>());

        if (uNode == carNode)
        {
            // carNode는 PC5(IF1)와 Uu(IF2) 둘 다 가진 유일한 노드.
            // Uu 게이트웨이 기본경로는 반드시 IF2(Uu)로 설정해야 한다 —
            // 기존 코드는 모든 노드에 IF1을 썼는데, carNode의 IF1은 PC5라서
            // Uu 방향 기본경로가 PC5 인터페이스로 잘못 나가고 있었다.
            ueStaticRouting->SetDefaultRoute(
                epcHelper->GetUeDefaultGatewayAddress(), 2);
        }
        else
        {
            ueStaticRouting->SetDefaultRoute(
                epcHelper->GetUeDefaultGatewayAddress(), 1);
        }
    }

    {
        Ptr<Ipv4StaticRouting> carRoutingDump =
            ipv4RoutingHelper.GetStaticRouting(carNode->GetObject<Ipv4>());
        std::cout << "======= carNode Routing Table =======" << std::endl;
        carRoutingDump->PrintRoutingTable(Create<OutputStreamWrapper>(&std::cout));
        std::cout << "=======================================" << std::endl;
    }

    {
        Ptr<Ipv4StaticRouting> carStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(carNode->GetObject<Ipv4>());
        // SL 인터페이스(IF1)를 기본 멀티캐스트 출력으로 설정
        carStaticRouting->SetDefaultMulticastRoute(1);
    }

    // Remote Host → UE 서브넷 (7.0.0.0/8) — wwan0(Uu) 경로
    // internetDevices = p2ph.Install(pgw, remoteHost):
    //   Get(0) = PGW 측 디바이스, Get(1) = RemoteHost 측 디바이스
    remoteHostStaticRouting->AddNetworkRouteTo(
        Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"),
        internetDevices.Get(1)->GetIfIndex());

    // Remote Host → 8.0.0.0/8 (v2v0/PC5 경로, PGW 경유)
    remoteHostStaticRouting->AddNetworkRouteTo(
        Ipv4Address("8.0.0.0"), Ipv4Mask("255.0.0.0"),
        internetDevices.Get(1)->GetIfIndex());
    // 10.0.x.0/24 RSU 백홀 경로는 bhDevices 루프에서 이미 추가됨

    // ----------------------------------------------------------
    // [추가] PGW v2v0 라우팅 (8.0.0.0/8 → TUN 인터페이스)
    //
    // EPC PGW는 AssignUeIpv4Address가 할당한 7.x.x.x 주소만 TUN으로 라우팅.
    // 수동 추가한 8.x.x.x 보조 주소도 동일 TUN(EPC 내부 경로)을 통해
    // 전달될 수 있도록 PGW TUN 인터페이스에 8.0.0.0/8 경로를 추가.
    // ----------------------------------------------------------
    {
        Ptr<Ipv4> pgwIpv4 = pgw->GetObject<Ipv4>();
        Ptr<Ipv4StaticRouting> pgwStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(pgwIpv4);

        // PGW의 TUN 인터페이스: 7.0.0.0/8 서브넷 주소를 가진 인터페이스 탐색
        // Ipv4Mask::IsMatch(a, b): a와 b가 동일 서브넷이면 true
        const Ipv4Mask mask8("255.0.0.0");
        uint32_t pgwTunIfIdx = 0;
        for (uint32_t i = 0; i < pgwIpv4->GetNInterfaces(); ++i)
        {
            for (uint32_t j = 0; j < pgwIpv4->GetNAddresses(i); ++j)
            {
                Ipv4Address addr = pgwIpv4->GetAddress(i, j).GetLocal();
                if (mask8.IsMatch(addr, Ipv4Address("7.0.0.1")))
                {
                    pgwTunIfIdx = i;
                }
            }
        }
        // TUN IF를 찾은 경우에만 추가 (못 찾으면 라우팅 없이 진행)
        if (pgwTunIfIdx > 0)
        {
            pgwStaticRouting->AddNetworkRouteTo(
                Ipv4Address("8.0.0.0"), Ipv4Mask("255.0.0.0"), pgwTunIfIdx);
        }

        // PGW → RemoteHost 서브넷 경로 (Uu 업링크 → RemoteHost 도달)
        // EPC helper는 이 경로를 자동으로 등록하지 않으므로 명시적 추가.
        pgwStaticRouting->AddNetworkRouteTo(
            Ipv4Address("1.0.0.0"), Ipv4Mask("255.0.0.0"),
            internetDevices.Get(0)->GetIfIndex());
    }

    // ----------------------------------------------------------
    // [추가] UE1 Static Route: 8.0.0.0/8 → IF1 (v2v0/PC5 인터페이스)
    //   ns-3는 AddAddress() 시 네트워크 경로를 자동 등록하지만,
    //   명시적으로 추가하여 Static Routing 우선순위를 보장.
    //   metric=1 (Uu 기본경로와 동일 우선순위, src IP로 구분)
    // ----------------------------------------------------------
    {
        Ptr<Ipv4StaticRouting> ue1StaticRouting =
            ipv4RoutingHelper.GetStaticRouting(carNode->GetObject<Ipv4>());
        // v2v0 네트워크 경로: 8.0.0.0/8 → IF1 (NrUeNetDevice)
        ue1StaticRouting->AddNetworkRouteTo(
            Ipv4Address("8.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

        // RSU도 v2v0 네트워크 경로 추가
        Ptr<Ipv4StaticRouting> rsuStaticRouting2 =
            ipv4RoutingHelper.GetStaticRouting(
                allSlUesContainer.Get(0)->GetObject<Ipv4>());
        rsuStaticRouting2->AddNetworkRouteTo(
            Ipv4Address("8.0.0.0"), Ipv4Mask("255.0.0.0"), 1);
    }
    // UE1(Car) IP 주소 획득 (ToD 수신 소켓 바인딩용)
    Ipv4Address ue1Addr = ueIpIface.GetAddress(numRsu + 1);

    // ----------------------------------------------------------
    // [추가] v2v0 (PC5) 인터페이스 보조 주소 할당 (8.x.x.x 대역)
    //
    // 이기종 통신 인터페이스 매핑:
    //   wwan0 = Uu  (5G gNB 경유)   → 7.x.x.x  (EPC 자동 할당)
    //   v2v0  = PC5 (SL 직접링크)   → 8.x.x.x  (수동 할당)
    //
    // 소스 IP 기반 경로 결정 원칙:
    //   src=7.x.x.x → wwan0/Uu  경로 → gNB → PGW → MEC → Remote
    //   src=8.x.x.x → v2v0/PC5  경로 → SL 직접링크 → RSU
    //
    // 구현 방식: 동일 NrUeNetDevice(IF1)에 보조 주소 추가
    //   실차 Linux OBU에서는 wwan0/v2v0 가 각각 별도 NIC이지만,
    //   ns-3 시뮬레이션에서는 NrUeNetDevice가 Uu+SL을 모두 처리하므로
    //   동일 인터페이스에 두 주소를 추가하여 논리적 경로 분리를 모델링.
    // ----------------------------------------------------------
    const Ipv4Address ue1V2v0Addr("8.0.0.2");   // UE1(Car) v2v0 IP
    const Ipv4Address rsuV2v0Addr("8.0.0.1");   // RSU0    v2v0 IP
    const Ipv4Address rsu1V2v0Addr("8.0.0.3");  // RSU1    v2v0 IP
    const Ipv4Address rsu2V2v0Addr("8.0.0.4");  // RSU2    v2v0 IP
    const Ipv4Address rsu3V2v0Addr("8.0.0.5");  // RSU3    v2v0 IP
    {
        // UE1(Car) v2v0 주소 추가
        Ptr<Ipv4> carIpv4  = carNode->GetObject<Ipv4>();
        uint32_t  carIfIdx = carIpv4->GetInterfaceForDevice(
                                 allSlUesNetDeviceContainer.Get(numRsu));
        carIpv4->AddAddress(carIfIdx,
            Ipv4InterfaceAddress(ue1V2v0Addr, Ipv4Mask("255.0.0.0")));

        // RSU0 v2v0 주소 추가
        Ptr<Ipv4> rsuIpv4n  = allSlUesContainer.Get(0)->GetObject<Ipv4>();
        uint32_t  rsuIfIdx2 = rsuIpv4n->GetInterfaceForDevice(
                                  allSlUesNetDeviceContainer.Get(0));
        rsuIpv4n->AddAddress(rsuIfIdx2,
            Ipv4InterfaceAddress(rsuV2v0Addr, Ipv4Mask("255.0.0.0")));

        // RSU1~3 v2v0 주소 추가 (numRsu=1인 현재 시나리오에는 해당 RSU가
        // 존재하지 않으므로 인덱스 1/2/3을 그대로 쓰면 carNode를 잘못
        // 덮어쓰거나(1) allSlUesContainer 범위를 벗어난다(2,3). numRsu가
        // 그만큼 클 때만 실행되도록 방어한다.
        if (numRsu > 1) {
            Ptr<Ipv4> rsu1Ipv4n  = allSlUesContainer.Get(1)->GetObject<Ipv4>();
            uint32_t  rsu1IfIdx  = rsu1Ipv4n->GetInterfaceForDevice(
                                       allSlUesNetDeviceContainer.Get(1));
            rsu1Ipv4n->AddAddress(rsu1IfIdx,
                Ipv4InterfaceAddress(rsu1V2v0Addr, Ipv4Mask("255.0.0.0")));
        }

        if (numRsu > 2) {
            Ptr<Ipv4> rsu2Ipv4n  = allSlUesContainer.Get(2)->GetObject<Ipv4>();
            uint32_t  rsu2IfIdx  = rsu2Ipv4n->GetInterfaceForDevice(
                                       allSlUesNetDeviceContainer.Get(2));
            rsu2Ipv4n->AddAddress(rsu2IfIdx,
                Ipv4InterfaceAddress(rsu2V2v0Addr, Ipv4Mask("255.0.0.0")));
        }

        if (numRsu > 3) {
            Ptr<Ipv4> rsu3Ipv4n  = allSlUesContainer.Get(3)->GetObject<Ipv4>();
            uint32_t  rsu3IfIdx  = rsu3Ipv4n->GetInterfaceForDevice(
                                       allSlUesNetDeviceContainer.Get(3));
            rsu3Ipv4n->AddAddress(rsu3IfIdx,
                Ipv4InterfaceAddress(rsu3V2v0Addr, Ipv4Mask("255.0.0.0")));
        }
    }

    // ----------------------------------------------------------
    // [ToD 소켓] server DL 송신 소켓 + 전역 주소 등록
    // ----------------------------------------------------------

    // DL 방향 목적지 주소 전역 저장 (SendTodPeriodic 에서 사용)
    g_ue1Addr     = ue1Addr;
    g_ue1V2v0Addr = ue1V2v0Addr;
    g_rsuV2v0Addr = rsuV2v0Addr;
    g_rsuUuAddr   = ueIpIface.GetAddress(0, 0);

    // server DL 송신 소켓 2개 (Uu / PC5 경로) — AddPath()는 g_uuPortGlobal 설정 후 호출
    Ptr<Socket> uuTxSock = Socket::CreateSocket(remoteHost, UdpSocketFactory::GetTypeId());
    uuTxSock->Bind(InetSocketAddress(Ipv4Address::GetAny(), 0));
    Ptr<Socket> pc5TxSock = Socket::CreateSocket(remoteHost, UdpSocketFactory::GetTypeId());
    pc5TxSock->Bind(InetSocketAddress(Ipv4Address::GetAny(), 0));
    g_serverGlobalAddr = remoteHostAddr;

    // ----------------------------------------------------------
    // [트래픽 2] ToD (원격제어): DL server→carNode (Uu 직접 or PC5 RSU 릴레이)
    //
    // 구조:
    //   Uu 모드 : server → carNode(ue1Addr)  직접 전송
    //   PC5 모드: server → RSU(rsuV2v0Addr) → carNode(ue1V2v0Addr) SL 릴레이
    // ----------------------------------------------------------

    // ① ToD 수신: carNode plain UDP — PacketSink + NotifyTodRx 트레이스
    uint16_t todPort = uuPort;
    {
        PacketSinkHelper todSink("ns3::UdpSocketFactory",
            InetSocketAddress(Ipv4Address::GetAny(), todPort));
        ApplicationContainer sinkApps = todSink.Install(carNode);
        sinkApps.Start(Seconds(0.0));
        sinkApps.Stop(Seconds(g_simEndTime));
        Ptr<PacketSink> sink = DynamicCast<PacketSink>(sinkApps.Get(0));
        if (sink)
            sink->TraceConnectWithoutContext("RxWithAddresses",
                MakeCallback(&NotifyTodRx));
        std::cout << ">>> carNode ToD PacketSink bound @ *:" << todPort << std::endl;
    }

    {
        Ptr<Socket> carPc5RecvSocket = Socket::CreateSocket(
            carNode, UdpSocketFactory::GetTypeId());
        carPc5RecvSocket->BindToNetDevice(
            allSlUesNetDeviceContainer.Get(numRsu));  // carNode 디바이스
        carPc5RecvSocket->Bind(
            InetSocketAddress(groupAddress4, slPort));
        carPc5RecvSocket->SetRecvCallback(
            MakeCallback(&NotifyUe1PcToD));
        std::cout << ">>> carNode PC5 relay RX socket bound @ SL:"
                  << todPort << std::endl;
    }

    // [DIAG] carNode IP 인터페이스 정보 (디버그용)
    {
        Ptr<Ipv4> ipv4Car = carNode->GetObject<Ipv4>();
        (void)ipv4Car;
    }


    // ② RSU 릴레이 설정 (numRsu개, 루프)
    for (uint32_t i = 0; i < numRsu; ++i) {
        Ptr<Node> rsuNode = allSlUesContainer.Get(i);

        Ptr<Socket> rsuRelayRxSock = Socket::CreateSocket(
            rsuNode, UdpSocketFactory::GetTypeId());
        rsuRelayRxSock->BindToNetDevice(bhDevices[i].Get(1));
        rsuRelayRxSock->Bind(
            InetSocketAddress(Ipv4Address::GetAny(), todPort));
        rsuRelayRxSock->SetRecvCallback(
            MakeCallback(&NotifyRsuRelayRx));

        Ptr<Socket> txSock = Socket::CreateSocket(
            rsuNode, UdpSocketFactory::GetTypeId());
        txSock->BindToNetDevice(allSlUesNetDeviceContainer.Get(i));
        txSock->SetAllowBroadcast(true);
        txSock->Connect(InetSocketAddress(groupAddress4, slPort));

        g_rsuRelayTxSockets.push_back(txSock);
        g_rsuRelaySocketMap[rsuNode->GetId()] = txSock;
        std::cout << ">>> RSU" << i << " relay TX socket → SL" << std::endl;
    }

    // ③ ToD 송신 주기 계산: SeqTsSizeHeader(20B) 오버헤드만 반영
    g_uuPortGlobal = todPort;

    // 경로 등록. path 0 = PRIMARY(Uu), path 1 = SECONDARY(PC5 relay).
    g_pathTx.AddPath(TVT_AI_PRIMARY_PATH, uuTxSock,
        InetSocketAddress(g_ue1Addr,   g_uuPortGlobal));   // Uu: server→carNode 직접
    g_pathTx.AddPath(TVT_AI_SECONDARY_PATH, pc5TxSock,
        InetSocketAddress(g_rsuBackhaulAddrs[0], g_uuPortGlobal));  // PC5: server→RSU0(백홀)→carNode 릴레이
    // 두 소켓은 여기서 한 번 등록된 뒤 시뮬레이션이 끝날 때까지 유지된다.
    // action 은 SECONDARY 에 중복 copy 를 얼마나 실을지만 고른다 — 어느 경로가
    // primary 인지는 고정 정책이며 학습 대상이 아니다.
    g_pathTx.SetAction(g_currentAction);
    // Episode 시작 시 credit = 0. 이전 episode 의 꼬리가 다음 episode 의 첫
    // packet 들을 결정하면 episode 독립성이 깨진다.
    g_pathTx.ResetAccumulator();
    g_dqnState.currentAction = g_currentAction;
    g_lastAiResult.applied = g_currentAction;

    SeqTsSizeHeader tmpHdr;
    uint32_t hdrBytes   = tmpHdr.GetSerializedSize();   // 20 B
    double   bitsPerPkt = (g_todPktSize + hdrBytes) * 8.0;
    Time     todInterval = Seconds(bitsPerPkt / (todDataRateMbps * 1e6));

    // 1.0s 이후 주기 발신 시작 (gNB attach 완료 후, DL: server 컨텍스트)
    // Traffic begins only after the observation warm-up has filled the
    // planner's history with real frames.
    g_trafficStartSec = g_decisionStartSec
        + g_observationWarmupSteps * (g_dqnDecisionIntervalMs / 1000.0);
    Simulator::ScheduleWithContext(remoteHost->GetId(),
                                   Seconds(g_trafficStartSec),
                                   &SendTodPeriodic, todInterval);
    std::cout << ">>> [WARMUP] decisions start " << g_decisionStartSec
              << "s, history warm-up " << g_observationWarmupSteps
              << " steps, traffic + formal accounting start "
              << g_trafficStartSec << "s" << std::endl;

    // ----------------------------------------------------------
    // [RTT 루프백] ACK 소켓 설정 (왕복 지연 측정) — DL 방향
    //   carNode: g_todAckSocket → NotifyTodRx 내에서 server로 ACK 즉시 송신
    //   server(remoteHost): g_mecAckRxSocket → g_ackPort 수신 → NotifyTodAckRx → RTT 계산
    // ----------------------------------------------------------
    g_todAckSocketUu = Socket::CreateSocket(carNode, UdpSocketFactory::GetTypeId());
    g_todAckSocketUu->BindToNetDevice(carUuNetDevContainer.Get(0));
    g_todAckSocketUu->Bind(InetSocketAddress(Ipv4Address::GetAny(), 0));

    // [제거됨] g_todAckSocketPc5 생성 + BindToNetDevice(carSlNetDevContainer, IF1).
    // ACK은 RAT과 무관하게 항상 위 g_todAckSocketUu로만 전송한다 — 사유는
    // g_todAckSocketUu 선언부 주석 참조 (RSU가 PC5 역방향 릴레이 미지원).
    std::cout << ">>> carNode ACK TX socket created (→ server:" << g_ackPort << ")" << std::endl;

    g_mecAckRxSocket = Socket::CreateSocket(remoteHost, UdpSocketFactory::GetTypeId());
    g_mecAckRxSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), g_ackPort));
    g_mecAckRxSocket->SetRecvCallback(MakeCallback(&NotifyTodAckRx));
    std::cout << ">>> server ACK RX socket bound @ *:" << g_ackPort << std::endl;

    std::cout << ">>> ToD traffic (plain UDP): "
              << todDataRateMbps << " Mbps"
              << " interval=" << todInterval.GetMilliSeconds() << "ms"
              << " Uu/PC5 (DDQN 선택)" << std::endl;

    // ----------------------------------------------------------
    // Uu RLC 수신 Trace
    // ----------------------------------------------------------
    // SL RLC 수신 DB 저장 (CP 통계용)
    // NotifyUuRlcPduRx는 제거: 동일 trace에 연결하면 SL PDU가 Uu 카운터에 이중 집계됨.
    // Uu 통계는 NotifyTodRx (app layer) 에서만 집계.
    Config::ConnectWithoutContext(
        "/NodeList/*/DeviceList/*/$ns3::NrUeNetDevice/"
        "ComponentCarrierMapUe/*/NrUeMac/RxRlcPduWithTxRnti",
        MakeBoundCallback(&NotifySlRlcPduRx, &g_rlcRxStats));

    // ----------------------------------------------------------
    // [추가] RSRP / SINR Trace 연결 (Uu 링크 품질 모니터링)
    // gNB attach 이후 활성화, 핸드오버 판단에 사용
    // ----------------------------------------------------------
    Config::Connect(
        "/NodeList/*/DeviceList/*/$ns3::NrUeNetDevice/"
        "ComponentCarrierMapUe/*/NrUePhy/DlCtrlSinr",
        MakeCallback(&SinrCallback));

    Config::Connect(
        "/NodeList/*/DeviceList/*/$ns3::NrUeNetDevice/"
        "ComponentCarrierMapUe/*/NrUePhy/ReportRsrp",
        MakeCallback(&RsrpCallback));

    // ----------------------------------------------------------
    // [추가] SL PSSCH 스케줄링 트레이스 연결 (HARQ 재전송 카운팅)
    // NrUeNetDevice가 MAC을 attribute로 노출하지 않아 Config::Connect 문자열
    // 경로를 쓸 수 없음 → GetMac()으로 직접 객체를 얻어 TraceConnect
    // ----------------------------------------------------------
    for (auto it = allSlUesNetDeviceContainer.Begin();
        it != allSlUesNetDeviceContainer.End(); ++it)
    {
        Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(*it);
        if (!ueDev) continue;

        for (uint32_t bIdx = 0; bIdx < ueDev->GetCcMapSize(); ++bIdx)
        {
            Ptr<NrUeMac> mac = ueDev->GetMac(bIdx);
            Ptr<NrSlUeMac> slMac = DynamicCast<NrSlUeMac>(mac);
            if (slMac)
            {
                // [수정] g_psschTxStats(DB 미연결) 제거에 따라 MakeBoundCallback
                // → MakeCallback으로 변경 (더 이상 bound 객체가 필요 없음)
                slMac->TraceConnectWithoutContext(
                    "SlPsschScheduling",
                    MakeCallback(&NotifySlPsschScheduling));

                // CBR: ego 의 SL MAC 에서만 수집한다. ego 가 자신이 겪는
                // 혼잡을 어떻게 보는지가 duplication 결정의 맥락이기 때문이다.
                // enablePc5Debug 와 무관하게 항상 켠다 — SC-3/SC-14 의 자원
                // 압력 축이 이 값에 의존한다. LOGGING ONLY.
                if (g_carNodePtr &&
                    ueDev->GetNode()->GetId() == g_carNodePtr->GetId())
                {
                    slMac->TraceConnectWithoutContext(
                        "CbrMeasurement", MakeCallback(&EgoCbrCallback));
                }

                // [RADIO-DIAG] RSU의 NrSlUeMacScheduler에만 Grant/RLC버퍼 계측
                // 트레이스 연결 (RSU→carNode ToD 릴레이를 만드는 스케줄러).
                // NrSlUeMacScheduler는 attribute(PointerValue)로 노출되어
                // 있으므로 GetAttribute()로 얻어 TraceConnectWithoutContext.
                // --enablePc5Debug 로 on/off (기본 false).
                if (g_rsuNodesGlobal.GetN() > 0 &&
                    ueDev->GetNode()->GetId() == g_rsuNodesGlobal.Get(0)->GetId())
                {
                    PointerValue schedPtrValue;
                    slMac->GetAttribute("NrSlUeMacScheduler", schedPtrValue);
                    Ptr<NrSlUeMacScheduler> sched = schedPtrValue.Get<NrSlUeMacScheduler>();
                    if (sched)
                    {
                        // Contention proxy — 항상 켠다 (LOGGING ONLY).
                        sched->TraceConnectWithoutContext(
                            "GrantCreated", MakeCallback(&SlTxGrantCreatedCallback));
                        sched->TraceConnectWithoutContext(
                            "GrantPublished", MakeCallback(&SlTxGrantPublishedCallback));
                    }
                    if (sched && g_enablePc5Debug)
                    {
                        sched->TraceConnectWithoutContext(
                            "GrantCreated", MakeCallback(&Pc5GrantCreatedCallback));
                        sched->TraceConnectWithoutContext(
                            "GrantPublished", MakeCallback(&Pc5GrantPublishedCallback));
                        sched->TraceConnectWithoutContext(
                            "RlcBufferStatus", MakeCallback(&Pc5RlcBufferStatusCallback));
                        std::cout << "[Trace] PC5 SL 스케줄러 계측(RSU) 연결 성공" << std::endl;
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------
    // [추가] PC5 PSSCH 수신 트레이스 연결 (PC5 SINR / RSRP 실측)
    //
    // RxPsschTraceUe: SlRxDataPacketTraceParams 콜백
    //   → SlPsschRxCallback()에서 m_sinr(선형)을 dB로 변환
    //   → g_slSinr_dB / g_slRsrp_dBm 갱신
    //
    // NrUeNetDevice/ComponentCarrierMapUe/* 는 Uu BWP 기준이므로
    // NrSpectrumPhy 레벨에서 연결해야 SL BWP 수신도 포착됨.
    // ----------------------------------------------------------
    Config::Connect(
        "/NodeList/*/DeviceList/*/$ns3::NrUeNetDevice/"
        "ComponentCarrierMapUe/*/NrUePhy/NrSpectrumPhy/RxPsschTraceUe",
        MakeCallback(&SlPsschRxCallback));

    // ----------------------------------------------------------
    // [추가] Uu/PC5 RSSI 직접 측정 트레이스 연결
    //
    // NrUeNetDevice가 NrInterference/NrSlInterference를 attribute로 노출하지
    // 않으므로 Config::Connect 문자열 경로를 쓸 수 없음 → carUuDev/carSlDev
    // 포인터로 직접 GetSpectrumPhy()->GetNrInterference()/GetNrSlInterference()
    // 를 얻어 TraceConnectWithoutContext (SlPsschScheduling과 동일한 패턴).
    // carUuNetDevContainer/carSlNetDevContainer 둘 다 CC 1개뿐이므로 GetPhy(0) 고정.
    // ----------------------------------------------------------
    {
        Ptr<NrUeNetDevice> carUuDevForRssi = DynamicCast<NrUeNetDevice>(carUuNetDevContainer.Get(0));
        bool uuConnected = false;
        if (carUuDevForRssi)
        {
            uuConnected = carUuDevForRssi->GetPhy(0)->GetSpectrumPhy()->GetNrInterference()
                ->TraceConnectWithoutContext("RssiPerProcessedChunk", MakeCallback(&UuRssiCallback));
        }
        std::cout << "[Trace] Uu RssiPerProcessedChunk 연결 "
                  << (uuConnected ? "성공" : "실패") << std::endl;

        Ptr<NrUeNetDevice> carSlDevForRssi = DynamicCast<NrUeNetDevice>(carSlNetDevContainer.Get(0));
        bool slConnected = false;
        if (carSlDevForRssi)
        {
            slConnected = carSlDevForRssi->GetPhy(0)->GetSpectrumPhy()->GetNrSlInterference()
                ->TraceConnectWithoutContext("SlRssiPerProcessedChunk", MakeCallback(&Pc5RssiCallback));
        }
        std::cout << "[Trace] PC5 SlRssiPerProcessedChunk 연결 "
                  << (slConnected ? "성공" : "실패") << std::endl;

        // [RADIO-DIAG] PC5 링크버짓/LOS/NLOSv/Shadowing 계측 — --enablePc5Debug 로 on/off.
        if (g_enablePc5Debug)
        {
            bool linkBudgetConnected = false;
            if (carSlDevForRssi)
            {
                linkBudgetConnected = carSlDevForRssi->GetPhy(0)->GetSpectrumPhy()->GetNrSlInterference()
                    ->TraceConnectWithoutContext("SlLinkBudget", MakeCallback(&Pc5LinkBudgetCallback));
            }
            std::cout << "[Trace] PC5 SlLinkBudget 연결 "
                      << (linkBudgetConnected ? "성공" : "실패") << std::endl;

            // [RADIO-DIAG] SL 대역(BWP)의 ChannelConditionModel에 LOS/NLOS/NLOSv 판정
            // 계측 트레이스 연결. bandSl은 numCcPerBand=1로 CC/BWP가 하나뿐이므로
            // bandSl.m_cc[0]->m_bwp[0]->m_propagation이 SL 전파모델 인스턴스이다.
            bool losConnected = false;
            if (!bandSl.m_cc.empty() && !bandSl.m_cc[0]->m_bwp.empty())
            {
                Ptr<PropagationLossModel> slPropLoss = bandSl.m_cc[0]->m_bwp[0]->m_propagation;
                Ptr<ThreeGppPropagationLossModel> slPropLoss3gpp =
                    DynamicCast<ThreeGppPropagationLossModel>(slPropLoss);
                if (slPropLoss3gpp)
                {
                    Ptr<ChannelConditionModel> ccm = slPropLoss3gpp->GetChannelConditionModel();
                    if (ccm)
                    {
                        losConnected = ccm->TraceConnectWithoutContext(
                            "LosConditionComputed", MakeCallback(&Pc5LosConditionCallback));
                    }
                }
            }
            std::cout << "[Trace] PC5 LosConditionComputed 연결 "
                      << (losConnected ? "성공" : "실패") << std::endl;

            // [RADIO-DIAG] NLOSv 추가손실 / Shadowing 손실 계측 트레이스 연결
            bool nlosvConnected = false;
            bool shadowConnected = false;
            if (!bandSl.m_cc.empty() && !bandSl.m_cc[0]->m_bwp.empty())
            {
                Ptr<PropagationLossModel> slPropLoss = bandSl.m_cc[0]->m_bwp[0]->m_propagation;
                Ptr<ThreeGppPropagationLossModel> slPropLoss3gpp =
                    DynamicCast<ThreeGppPropagationLossModel>(slPropLoss);
                if (slPropLoss3gpp)
                {
                    shadowConnected = slPropLoss3gpp->TraceConnectWithoutContext(
                        "ShadowingApplied", MakeCallback(&Pc5ShadowingCallback));
                }
                Ptr<ThreeGppV2vUrbanPropagationLossModel> slPropLossV2v =
                    DynamicCast<ThreeGppV2vUrbanPropagationLossModel>(slPropLoss);
                if (slPropLossV2v)
                {
                    nlosvConnected = slPropLossV2v->TraceConnectWithoutContext(
                        "NlosvAdditionalLoss", MakeCallback(&Pc5NlosvLossCallback));
                }
            }
            std::cout << "[Trace] PC5 NlosvAdditionalLoss 연결 "
                      << (nlosvConnected ? "성공" : "실패") << std::endl;
            std::cout << "[Trace] PC5 ShadowingApplied 연결 "
                      << (shadowConnected ? "성공" : "실패") << std::endl;
        }
    }

    // [진단용] Uu(원격제어) 채널의 ChannelConditionModel LOS/NLOS/NLOSv 판정
    // 계측. PC5 진단과는 무관한 별개 인터페이스이므로 g_enablePc5Debug에
    // 종속시키지 않고 항상 연결한다.
    {
        bool uuLosConnected = false;
        if (!bandUu.m_cc.empty() && !bandUu.m_cc[0]->m_bwp.empty())
        {
            Ptr<PropagationLossModel> uuPropLoss = bandUu.m_cc[0]->m_bwp[0]->m_propagation;
            Ptr<ThreeGppPropagationLossModel> uuPropLoss3gpp =
                DynamicCast<ThreeGppPropagationLossModel>(uuPropLoss);
            if (uuPropLoss3gpp)
            {
                Ptr<ChannelConditionModel> uuCcm = uuPropLoss3gpp->GetChannelConditionModel();
                if (uuCcm)
                {
                    uuLosConnected = uuCcm->TraceConnectWithoutContext(
                        "LosConditionComputed", MakeCallback(&UuLosConditionCallback));
                }
            }
        }
        std::cout << "[Trace] Uu LosConditionComputed 연결 "
                  << (uuLosConnected ? "성공" : "실패") << std::endl;
    }

    // ----------------------------------------------------------
    // [추가] V2X 상태 수집 루프 시작 (10ms 주기, gNB attach 후)
    //   CollectV2xState() → Python DDQN → QuicPathManager 경로 전환
    //   보상 R = 0.5×R_lat + 0.25×R_ho + 0.15×R_qual + 0.10×R_loss
    // ----------------------------------------------------------
    Simulator::Schedule(Seconds(g_decisionStartSec), &CollectV2xState);
    std::cout << ">>> [DDQN] V2xState 수집 루프 시작 (10ms, t=1.5s~)"
              << std::endl;

    // ── LEGACY_ONLY 시나리오 CSV (기본 OFF) ───────────────────────────────
    if (g_enableLegacyHoTelemetry)
    {
        std::cout << "[Scenario][LEGACY] 출력 디렉토리: " << g_scenOutDir << "\n";
        g_csvS1.open(g_scenOutDir + "tvt_scenario1.csv");
        g_csvS1 << "handover_index,dqn_inference_delay_ms,interface_switching_delay_ms,total_delay_ms\n";
        g_csvS2.open(g_scenOutDir + "tvt_scenario2.csv");
        g_csvS2 << "handover_index,avg_handover_delay_ms,avg_non_handover_delay_ms,difference_ms\n";
        g_csvS3.open(g_scenOutDir + "tvt_scenario3.csv");
        g_csvS3 << "handover_index,delay1_ipc_ms,delay2_dqn_ms,delay3_switch_ms,total_delay_ms\n";
    }

    // ----------------------------------------------------------
    // DB 설정
    // ----------------------------------------------------------
    std::string exampleName = simTag + "-" + "nr-v2x-gnbrsu";
    g_db = std::make_unique<SQLiteOutput>(outputDir + exampleName + ".db");
    g_rlcRxStats.SetDb(g_db.get(), "rlcRx");

    // ----------------------------------------------------------
    // 이동성 기록
    // ----------------------------------------------------------
    std::string mobilityFileName = simTag + "-mobility.txt";
    if (generateInitialPosGnuScript)
        PrintUeInitPosToFile(simTag + "-init-pos.txt");
    RecordMobility(true, mobilityFileName);

    // ----------------------------------------------------------
    // [추가] 수직 핸드오버 판단 루프 시작 (100ms 주기)
    // gNB attach 완료 예상 시간(~1s) 이후 시작
    // ----------------------------------------------------------
    Simulator::Schedule(Seconds(1.5), &CheckAndSwitchInterface);
    Simulator::Schedule(Seconds(1.5), &UpdatePc5Path);
    std::cout << ">>> Vertical HO monitor scheduled (100ms period, starts at 1.5s)"
              << std::endl;

    // 두 경로를 모두 측정하기 위한 generic probe. 이전 구현에서는 비활성
    // 경로 하나만 프로브했고 gNB DL 충돌 우려로 한때 꺼져 있었으나, 구체적
    // 크래시 로그는 남아 있지 않다. 이제는 모든 duplication degree 에서 두
    // 경로 상태가 stale 해지지 않아야 하므로 항상 켠다.
    Simulator::Schedule(Seconds(1.5), &ProbeBothPaths);
    std::cout << ">>> ProbeBothPaths 활성화 (양쪽 경로, 100ms 주기, 1.5초 시작)"
              << std::endl;

    Simulator::Schedule(Seconds(1.5), &CheckPacketTimeout);
    std::cout << ">>> MP-QUIC timeout detector scheduled (100ms period, starts at 1.5s)"
              << std::endl;

    // BSM 제거 (GPS 브로드캐스팅으로 대체 예정)
    // Simulator::Schedule(Seconds(1.5), &SendSlPacket);

    // ----------------------------------------------------------
    // Throughput 주기 출력
    // ----------------------------------------------------------
    Simulator::Schedule(Seconds(1.0), &PrintThroughput);
    if (enablePc5Debug)
    {
        Simulator::Schedule(Seconds(1.0), &PrintPc5SchedulerStats);  // [RADIO-DIAG]
        Simulator::Schedule(Seconds(1.0), &PrintPc5LinkBudgetStats); // [RADIO-DIAG]
    }


    // ----------------------------------------------------------
    // ns3-ai 공유 메모리 연결 (실패 시 action=0 유지)
    // 시뮬레이션 시작 전에 `python3 tvt_test.py` 를 실행해야 한다.
    // ratMode가 uu/pc5(고정 모드)면 Python 프로세스 없이 실행 가능해야
    // 하므로 연결 자체를 시도하지 않는다.
    // ----------------------------------------------------------
    if (ratMode == "ddqn")
    {
        ConnectAiInterface();
    }
    else
    {
        std::cout << "[ratMode] '" << ratMode << "' 고정 모드 — ns3-ai 연결 생략" << std::endl;
    }

    // ----------------------------------------------------------
    // 시뮬레이션 실행
    // ----------------------------------------------------------
    Simulator::Stop(simTime);
    Simulator::Run();
    PrintFinalSummary();

    {
        std::string suffix = (ratMode != "ddqn") ? ("_" + ratMode) : ("_ho_" + startDup);
        // 시드 스윕에서 같은 모드의 런끼리 서로 덮어쓰지 않도록 rngRun을 붙인다.
        // rngRun==1(기존 동작)에서는 파일명을 바꾸지 않아 기존 스크립트가 그대로 동작한다.
        std::string runSuffix = (rngRun == 1) ? "" : ("_run" + std::to_string(rngRun));
        suffix += runSuffix;
        std::ofstream delayCsv(g_scenOutDir + "tvt_packet_delay" + suffix + ".csv");
        delayCsv << "packet_index,delay_ms,rx_time_s\n";
        for (auto& p : g_packetDelayLog)
            delayCsv << p.seq << "," << p.delayMs << "," << p.rxTimeSec << "\n";
        delayCsv.close();

        // LEGACY_ONLY: duplication-level 변경 시각 로그. 파일명은 과거
        // 아카이브와 맞추기 위해 유지한다(물리 handover 를 뜻하지 않는다).
        if (g_enableLegacyHoTelemetry && ratMode == "ddqn") {
            std::ofstream hoCsv(g_scenOutDir + "tvt_ho_delay_" + startDup + runSuffix + ".csv");
            hoCsv << "ho_index,ho_delay_ms\n";
            for (auto& p : g_hoDelayLog) hoCsv << p.first << "," << p.second << "\n";
            hoCsv.close();

            // 핸드오버 구간 판별용: 실제 전환이 일어난 시뮬레이션 시각.
            std::ofstream hoTimeCsv(g_scenOutDir + "tvt_ho_times_" + startDup + runSuffix + ".csv");
            hoTimeCsv << "ho_index,switch_time_s,direction\n";
            for (size_t i = 0; i < g_hoSwitchTimes.size(); ++i)
                hoTimeCsv << (i + 1) << "," << g_hoSwitchTimes[i].first << ","
                          << g_hoSwitchTimes[i].second << "\n";
            hoTimeCsv.close();
        }
    }

    // ----------------------------------------------------------
    // 시뮬레이션 종료 후 최종 통계
    // ----------------------------------------------------------
    double totalTime  = Simulator::Now().GetSeconds();
    double avgDelayMsFinal = (g_delayCount > 0) ? g_delaySumMs / static_cast<double>(g_delayCount) : 0.0;
    double lossRateFinal2  = (g_txTodPackets > 0)
        ? std::max(0.0, 1.0 - static_cast<double>(g_rxTodPackets) / static_cast<double>(g_txTodPackets))
        : 0.0;

    std::cout << ">>> [FINAL] Total Tx Packets   = " << g_txTodPackets << std::endl;
    std::cout << ">>> [FINAL] Total Rx Packets   = " << g_rxTodPackets << std::endl;
    std::cout << ">>> [FINAL] Avg E2E Delay      = " << std::fixed << std::setprecision(3) << avgDelayMsFinal << " ms" << std::endl;
    std::cout << ">>> [FINAL] Max E2E Delay      = " << std::setprecision(3) << g_delayMaxMs << " ms" << std::endl;
    std::cout << ">>> [FINAL] Packet Loss Rate   = " << std::setprecision(3) << (lossRateFinal2 * 100.0) << " %" << std::endl;
    // g_switchCount counts DUPLICATION-LEVEL CHANGES. It is not a handover
    // count: no bearer is re-established, no signalling is exchanged, no queue
    // is flushed and no interruption is modelled -- one integer changes. It is
    // a RETIRED diagnostic, never a state feature and never an objective term.
    // The name survives only because switch_count / switch_rate are already
    // columns in the archived CSVs and in the env contract's is_switch.
    std::cout << ">>> [FINAL] Action switch count = " << g_switchCount << std::endl;
    std::cout << ">>> [FINAL] action_hold_enabled = "
              << (kActionHoldEnabled ? 1 : 0) << std::endl;

    // [Time on Uu/PC5] 마지막으로 유지된 인터페이스의 남은 구간을 반영해 마감.
    {
        double elapsedFinal = totalTime - g_lastIfChangeAccountTime;
        if (elapsedFinal > 0.0)
        {
            if (g_currentAction >= 0 && g_currentAction < TVT_AI_ACTION_COUNT)
                g_timeOnLevelSec[g_currentAction] += elapsedFinal;
            g_lastIfChangeAccountTime = totalTime;
        }
    }
    for (int lvl = 0; lvl < TVT_AI_ACTION_COUNT; ++lvl)
        std::cout << ">>> [FINAL] Time on " << TvtActionName(lvl) << " = "
                  << std::fixed << std::setprecision(3)
                  << g_timeOnLevelSec[lvl] << " s" << std::endl;
    for (int lvl = 0; lvl < TVT_AI_ACTION_COUNT; ++lvl)
        std::cout << ">>> [FINAL] " << TvtActionName(lvl) << " selected count = "
                  << g_levelSelectedCount[lvl] << std::endl;
    std::cout << ">>> [FINAL] Cohorts opened     = " << g_cohortsOpened
              << "  published = " << g_cohortsPublished
              << "  unpublished_at_end = " << g_tickCohorts.size()
              << "  evicted = " << g_cohortsDropped << std::endl;
    std::cout << ">>> [FINAL] Logical Tx packets = " << g_logicalTxPackets << std::endl;
    std::cout << ">>> [FINAL] Logical Rx packets = " << g_logicalRxPackets << std::endl;
    std::cout << ">>> [FINAL] Attempted copies Tx = " << g_attemptedCopiesTx << std::endl;
    std::cout << ">>> [FINAL] Successful copies Tx = " << g_successfulCopiesTx << std::endl;
    std::cout << ">>> [FINAL] Duplicate copies Rx = " << g_duplicateCopiesRx << std::endl;
    std::cout << ">>> [FINAL] Duplicate bytes Rx = " << g_duplicateBytesRx << std::endl;
    {
        double attemptedPer = (g_logicalTxPackets > 0)
            ? static_cast<double>(g_attemptedCopiesTx) / g_logicalTxPackets : 0.0;
        double successfulPer = (g_logicalTxPackets > 0)
            ? static_cast<double>(g_successfulCopiesTx) / g_logicalTxPackets : 0.0;
        // DIAGNOSTIC ONLY (KNOWN_RISK_USEFUL_THROUGHPUT_PROXY). The reward reads
    // max(uu_bps, pc5_bps), a per-path byte rate that counts duplicate
    // arrivals too. These lines expose the direct logical delivery counts so
    // the gap between the proxy and true logical goodput is measurable. No
    // reward input is changed by printing them.
    {
        const double trafficSec = std::max(1e-9, g_simEndTime - g_trafficStartSec);
        const double bytesPerPkt = static_cast<double>(g_todPktSize) + 20.0;
        std::cout << ">>> [GOODPUT-DIAG] uu_arrivals=" << g_rxPacketsUu
                  << "  pc5_arrivals=" << g_rxPacketsSl
                  << "  logical_rx=" << g_logicalRxPackets
                  << "  duplicate_rx=" << g_duplicateCopiesRx
                  << "  traffic_window_s=" << std::fixed << std::setprecision(3)
                  << trafficSec
                  << "  direct_logical_goodput_bps=" << std::setprecision(1)
                  << (g_logicalRxPackets * bytesPerPkt * 8.0 / trafficSec)
                  << "  uu_path_rate_bps="
                  << (g_rxPacketsUu * bytesPerPkt * 8.0 / trafficSec)
                  << "  pc5_path_rate_bps="
                  << (g_rxPacketsSl * bytesPerPkt * 8.0 / trafficSec)
                  // 독립 검산: byte 누적기 vs packet 카운터 재구성.
                  // 두 값은 서로 다른 변수에서 나온다 — 하나는 도착마다 더한
                  // payload byte 합이고, 다른 하나는 logical packet 수에
                  // 공칭 payload 크기를 곱한 값이다.
                  << "  goodput_bytes_total=" << g_logicalGoodputBytesTotal
                  << "  goodput_from_bytes_bps="
                  << (g_logicalGoodputBytesTotal * 8.0 / trafficSec)
                  << "  goodput_from_counter_bps="
                  << (g_logicalRxPackets * static_cast<double>(g_todPktSize)
                      * 8.0 / trafficSec)
                  << std::endl;
    }
    std::cout << ">>> [FINAL] Attempted copies per logical = "
                  << std::setprecision(4) << attemptedPer << std::endl;
        std::cout << ">>> [FINAL] Successful copies per logical = "
                  << std::setprecision(4) << successfulPer << std::endl;
    }
    std::cout << ">>> [FINAL] Invalid action count = " << g_invalidActionCount << std::endl;
    if (g_minRsuCarDist < 1e18)
    {
        std::cout << ">>> [FINAL] Minimum vehicle-RSU distance = " << std::setprecision(3)
                  << g_minRsuCarDist << " m" << std::endl;
        std::cout << ">>> [FINAL] Time of minimum distance     = " << std::setprecision(3)
                  << g_minRsuCarDistTime << " s" << std::endl;
    }
    else
    {
        std::cout << ">>> [FINAL] Minimum vehicle-RSU distance = N/A" << std::endl;
    }

    // [RADIO-DIAG] A/B/C 링크버짓 실험용 전체 런 통계 (신호/SINR avg/min/max, LOS/NLOS/NLOSv 횟수)
    // --enablePc5Debug 로 on/off (기본 false) — 관련 트레이스가 연결된 경우에만 유의미.
    if (g_enablePc5Debug)
    {
        double sigAvg  = (g_pc5RunSampleCount > 0) ? g_pc5RunSignalSum / g_pc5RunSampleCount : NAN;
        double sinrAvg = (g_pc5RunSampleCount > 0) ? g_pc5RunSinrSum   / g_pc5RunSampleCount : NAN;
        double effLossAvg = PC5_RSU_TXPOWER_DBM - sigAvg;
        double effLossMax = PC5_RSU_TXPOWER_DBM - g_pc5RunSignalMin; // 최소 신호 = 최대 손실
        std::cout << std::fixed << std::setprecision(3)
                  << ">>> [FINAL] PC5 Signal avg/min/max = " << sigAvg << " / "
                  << g_pc5RunSignalMin << " / " << g_pc5RunSignalMax << " dBm" << std::endl;
        std::cout << ">>> [FINAL] PC5 SINR avg/min/max   = " << sinrAvg << " / "
                  << g_pc5RunSinrMin << " / " << g_pc5RunSinrMax << " dB" << std::endl;
        std::cout << ">>> [FINAL] PC5 EffLoss avg/max    = " << effLossAvg << " / "
                  << effLossMax << " dB" << std::endl;
        std::cout << ">>> [FINAL] LOS/NLOS/NLOSv count   = " << g_losCount << " / "
                  << g_nlosCount << " / " << g_nlosvCount << std::endl;

        uint64_t losTotal = g_losCount + g_nlosCount + g_nlosvCount;
        double losRatio = (losTotal > 0) ? 100.0 * g_losCount / losTotal : NAN;
        double nlosRatio = (losTotal > 0) ? 100.0 * g_nlosCount / losTotal : NAN;
        double nlosvRatio = (losTotal > 0) ? 100.0 * g_nlosvCount / losTotal : NAN;
        std::cout << ">>> [FINAL] LOS/NLOS/NLOSv ratio(%) = " << std::setprecision(1) << losRatio
                  << " / " << nlosRatio << " / " << nlosvRatio << std::endl;

        // [RADIO-DIAG] PSCCH-like(좁은 대역)/PSSCH-like(넓은 대역) 분리 통계 — 섞임 검증
        double narrowAvg = (g_narrowSampleCount > 0) ? g_narrowSignalSum / g_narrowSampleCount : NAN;
        double wideAvg    = (g_wideSampleCount   > 0) ? g_wideSignalSum   / g_wideSampleCount   : NAN;
        std::cout << std::setprecision(3)
                  << ">>> [FINAL] Narrow(<=" << kNarrowRbThreshold << "RB) n/avg/min/max = "
                  << g_narrowSampleCount << " / " << narrowAvg << " / " << g_narrowSignalMin
                  << " / " << g_narrowSignalMax << " dBm" << std::endl;
        std::cout << ">>> [FINAL] Wide(>" << kNarrowRbThreshold << "RB) n/avg/min/max    = "
                  << g_wideSampleCount << " / " << wideAvg << " / " << g_wideSignalMin
                  << " / " << g_wideSignalMax << " dBm" << std::endl;

        // [RADIO-DIAG] NLOSv 추가손실 / Shadowing 손실 — 별도 집계
        double nlosvLossAvg = (g_nlosvLossCount > 0) ? g_nlosvLossSum / g_nlosvLossCount : NAN;
        double shadowingAvg = (g_shadowingCount > 0) ? g_shadowingSum / g_shadowingCount : NAN;
        std::cout << ">>> [FINAL] NLOSv AddLoss n/avg/min/max = " << g_nlosvLossCount << " / "
                  << nlosvLossAvg << " / " << g_nlosvLossMin << " / " << g_nlosvLossMax
                  << " dB" << std::endl;
        std::cout << ">>> [FINAL] Shadowing n/avg/min/max     = " << g_shadowingCount << " / "
                  << shadowingAvg << " / " << g_shadowingMin << " / " << g_shadowingMax
                  << " dB" << std::endl;
    }

    // ── 경로별 측정 상태 (양 경로 모두 유지) ──────────────────────────
    {
        double nowSec = Simulator::Now().GetSeconds();
        for (int i = 0; i < 2; ++i)
        {
            const char* nm = (i == 0) ? "Uu " : "PC5";
            std::cout << ">>> [PATH-MEAS] path" << i << "(" << nm << ")"
                      << "  valid=" << (g_pathMeas[i].valid ? 1 : 0)
                      << "  rssi=" << std::fixed << std::setprecision(2)
                      << g_pathMeas[i].rssiDbm << "dBm"
                      << "  delay=" << g_pathMeas[i].delayMs << "ms"
                      << "  loss=" << std::setprecision(4) << g_pathLoss[i].Rate()
                      << "  thpt=" << std::setprecision(1) << g_pathMeas[i].thptBps << "bps"
                      << "  ageMs=" << std::setprecision(1) << g_pathMeas[i].AgeMs(nowSec)
                      << "  txCopies=" << g_pathTx.m_paths[i].txCopies
                      << "  txBytes=" << g_pathTx.m_paths[i].txBytes
                      << std::endl;
        }
    }


    // 위 요약 블록은 한 번만 출력한다. 예전에는 동일한 139줄이 두 번 실행되어
    // 모든 [FINAL] 값이 두 번 찍혔다. 두 번째 실행은 g_lastIfChangeAccountTime
    // 가드 때문에 elapsedFinal==0 이 되어 g_timeOnLevelSec 를 이중 계산하지
    // 않았으므로 숫자는 같았고, 이번 병합도 값을 바꾸지 않는다 — 출력만 한 번이
    // 된다. 아래는 원래 두 번째 블록에만 있던 고유 꼬리다.
    // ── [SCENARIO-DIAG] TVT_CONTENDED_CORRIDOR_V1 계측 요약 ───────────
    // LOGGING ONLY. 아래 어떤 값도 observation / reward / action 으로 가지 않는다.
    {
        const double cbrAvg = (g_cbrSamples > 0) ? g_cbrSum / g_cbrSamples : -1.0;
        std::cout << ">>> [SCENARIO-DIAG]"
                  << " active_veh=" << numActiveVeh
                  << " cam_tx_vehicles=" << camTxCount
                  << " cam_attempted=" << g_camAttempted
                  << " cam_tx_ok=" << g_camTxOk
                  << " cam_tx_fail=" << g_camTxFail
                  << " cam_suppressed=" << g_camSuppressed
                  << std::endl;
        std::cout << ">>> [SCENARIO-DIAG] sl_relay_contention_proxy"
                  << " grants=" << g_slTxGrants
                  << " reselections=" << g_slTxReselect
                  << " tb_published=" << g_slTxTbPublished
                  << " tb_bytes=" << g_slTxTbBytes
                  << "   (NOT CBR: this counts the SL relay's grant activity, "
                     "not pool occupancy)"
                  << std::endl;
        std::cout << ">>> [SCENARIO-DIAG] ego_cbr"
                  << " samples=" << g_cbrSamples
                  << " avg=" << std::fixed << std::setprecision(4)
                  << cbrAvg
                  << " min=" << ((g_cbrSamples > 0) ? g_cbrMin : -1.0)
                  << " max=" << ((g_cbrSamples > 0) ? g_cbrMax : -1.0)
                  << " last=" << g_cbrLast
                  << std::endl;
        std::cout << ">>> [SCENARIO-DIAG] load_transitions_s=";
        for (size_t i = 0; i < g_loadTransitions.size(); ++i)
            std::cout << std::setprecision(3) << g_loadTransitions[i]
                      << (i + 1 < g_loadTransitions.size() ? "," : "");
        if (g_loadTransitions.empty()) std::cout << "none";
        std::cout << "  rng_run=" << RngSeedManager::GetRun()
                  << "  mobility_offset_s=" << mobilityStartOffsetS
                  << std::endl;

        // active vehicle 공간 분포 (SC-1 검증용)
        if (numActiveVeh > 0 && g_carNodePtr)
        {
            Ptr<MobilityModel> egoMob = g_carNodePtr->GetObject<MobilityModel>();
            double dMin = 1e18, dMax = -1e18, dSum = 0.0;
            uint32_t within300 = 0;
            for (uint32_t i = 0; i < numActiveVeh; ++i)
            {
                Ptr<MobilityModel> m =
                    allSlUesContainer.Get(numRsu + 1 + i)->GetObject<MobilityModel>();
                if (!m || !egoMob) continue;
                const double d = CalculateDistance(m->GetPosition(), egoMob->GetPosition());
                dMin = std::min(dMin, d); dMax = std::max(dMax, d); dSum += d;
                if (d <= 300.0) ++within300;
            }
            std::cout << ">>> [GEOMETRY-PROBE] ego_to_active_at_end"
                      << " n=" << numActiveVeh
                      << " min=" << std::setprecision(1) << dMin
                      << " mean=" << (dSum / numActiveVeh)
                      << " max=" << dMax
                      << " within_300m=" << within300
                      << std::endl;
        }
    }

    // ── 논리/물리 패킷 회계 요약 ─────────────────────────────────────
    // duplication 이 logical goodput 을 부풀리지 않았는지 바로 확인할 수 있다:
    // logical_rx <= logical_tx 이어야 하고, duplicate 는 별도 항목으로만
    // 나타나야 한다.
    {
        double attemptedPer = (g_logicalTxPackets > 0)
            ? static_cast<double>(g_attemptedCopiesTx) / g_logicalTxPackets : 0.0;
        double successfulPer = (g_logicalTxPackets > 0)
            ? static_cast<double>(g_successfulCopiesTx) / g_logicalTxPackets : 0.0;
        std::cout << ">>> [LOGICAL-ACCT]"
                  << "  logical_tx="   << g_logicalTxPackets
                  << "  logical_rx="   << g_logicalRxPackets
                  << "  deadline_ok="  << g_logicalDeadlineSuccess
                  << "  deadline_miss=" << g_logicalDeadlineMiss
                  << "  attempted_copies_tx=" << g_attemptedCopiesTx
                  << "  successful_copies_tx=" << g_successfulCopiesTx
                  << "  duplicated_logical_packets=" << g_duplicatedLogicalPackets
                  << "  duplicate_rx=" << g_duplicateCopiesRx
                  << "  late_duplicate_rx=" << g_lateDuplicateRx
                  << "  duplicate_bytes=" << g_duplicateBytesRx
                  << "  attempted_per_logical=" << std::fixed << std::setprecision(4)
                  << attemptedPer
                  << "  successful_per_logical=" << successfulPer
                  << std::endl;
    }

    // ns3-ai: Python 에 종료 신호 전송 후 인터페이스 해제
    SendAiDone();

    // 메모리 정리
    g_rsuRelayTxSockets.clear();
    g_rsuRelaySocketMap.clear();
    g_mecAckRxSocket    = nullptr;
    g_todAckSocketUu    = nullptr;

    // ── 시나리오 CSV 닫기 및 그래프 생성 ───────────────────────────────────
    if (g_enableLegacyHoTelemetry)
    {
        g_csvS1.close();
        g_csvS2.close();
        g_csvS3.close();
        std::cout << "\n[Scenario][LEGACY] CSV 저장 완료: "
                     "tvt_scenario1.csv  tvt_scenario2.csv  tvt_scenario3.csv\n";
    }

    Simulator::Destroy();
    std::cout << "\n=== NS-3 Simulation finished successfully ===" << std::endl;

    if (!saveDb)
        std::remove((outputDir + exampleName + ".db").c_str());

    return 0;
}
