#include "tvt-ai.h"

#include <ns3/ns3-ai-msg-interface.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(ns3ai_tvt_py, m)
{
    m.attr("CONTRACT_VERSION") = ns3::TVT_AI_CONTRACT_VERSION;
    m.attr("ACTION_HOLD_ENABLED") = ns3::TVT_AI_ACTION_HOLD_ENABLED;
    m.attr("NO_MACRO") = ns3::TVT_AI_NO_MACRO;
    m.attr("ACTION_DUP_0") = static_cast<int32_t>(ns3::TVT_AI_ACTION_DUP_0);
    m.attr("ACTION_DUP_25") = static_cast<int32_t>(ns3::TVT_AI_ACTION_DUP_25);
    m.attr("ACTION_DUP_50") = static_cast<int32_t>(ns3::TVT_AI_ACTION_DUP_50);
    m.attr("ACTION_DUP_75") = static_cast<int32_t>(ns3::TVT_AI_ACTION_DUP_75);
    m.attr("ACTION_DUP_100") = static_cast<int32_t>(ns3::TVT_AI_ACTION_DUP_100);
    m.attr("ACTION_COUNT") = static_cast<int32_t>(ns3::TVT_AI_ACTION_COUNT);
    // The authoritative C++ action -> ratio table, exported so Python compares
    // against the real thing rather than against a transcribed copy.
    m.attr("DUPLICATION_LEVELS") = std::vector<double>(
        std::begin(ns3::TVT_AI_DUPLICATION_LEVELS),
        std::end(ns3::TVT_AI_DUPLICATION_LEVELS));
    m.def("duplication_ratio", &ns3::TvtAiDuplicationRatio);
    m.attr("PRIMARY_PATH") = ns3::TVT_AI_PRIMARY_PATH;
    m.attr("SECONDARY_PATH") = ns3::TVT_AI_SECONDARY_PATH;
    m.def("env_struct_size", []() { return sizeof(ns3::TvtAiEnv); });
    m.def("act_struct_size", []() { return sizeof(ns3::TvtAiAct); });

    py::class_<ns3::TvtAiEnv>(m, "PyEnvStruct")
        .def(py::init<>())
        .def_readwrite("contract_version", &ns3::TvtAiEnv::contract_version)
        .def_readwrite("simulation_time_s", &ns3::TvtAiEnv::simulation_time_s)

        .def_readwrite("uu_rssi", &ns3::TvtAiEnv::uu_rssi)
        .def_readwrite("pc5_rssi", &ns3::TvtAiEnv::pc5_rssi)
        .def_readwrite("uu_delay_ms", &ns3::TvtAiEnv::uu_delay_ms)
        .def_readwrite("pc5_delay_ms", &ns3::TvtAiEnv::pc5_delay_ms)
        .def_readwrite("uu_loss_rate", &ns3::TvtAiEnv::uu_loss_rate)
        .def_readwrite("pc5_loss_rate", &ns3::TvtAiEnv::pc5_loss_rate)
        .def_readwrite("uu_throughput_bps", &ns3::TvtAiEnv::uu_throughput_bps)
        .def_readwrite("pc5_throughput_bps", &ns3::TvtAiEnv::pc5_throughput_bps)
        .def_readwrite("uu_measurement_age_ms",
                       &ns3::TvtAiEnv::uu_measurement_age_ms)
        .def_readwrite("pc5_measurement_age_ms",
                       &ns3::TvtAiEnv::pc5_measurement_age_ms)
        .def_readwrite("uu_measurement_valid",
                       &ns3::TvtAiEnv::uu_measurement_valid)
        .def_readwrite("pc5_measurement_valid",
                       &ns3::TvtAiEnv::pc5_measurement_valid)

        // ---- mobility context c_t (contract 8) -----------------------
        // Upper-tier only. Exposed read/write so the regression suite can
        // construct synthetic envs, exactly like every other field.
        .def_readwrite("ego_speed_mps", &ns3::TvtAiEnv::ego_speed_mps)
        .def_readwrite("ego_rsu_distance_m",
                       &ns3::TvtAiEnv::ego_rsu_distance_m)

        .def_readwrite("current_action", &ns3::TvtAiEnv::current_action)
        .def_readwrite("action_hold_enabled",
                       &ns3::TvtAiEnv::action_hold_enabled)

        .def_readwrite("decision_id", &ns3::TvtAiEnv::decision_id)
        .def_readwrite("applied_decision_id",
                       &ns3::TvtAiEnv::applied_decision_id)
        .def_readwrite("requested_action", &ns3::TvtAiEnv::requested_action)
        .def_readwrite("applied_action", &ns3::TvtAiEnv::applied_action)
        .def_readwrite("apply_status", &ns3::TvtAiEnv::apply_status)
        .def_readwrite("transition_valid", &ns3::TvtAiEnv::transition_valid)

        .def_readwrite("e2e_delay_ms", &ns3::TvtAiEnv::e2e_delay_ms)
        .def_readwrite("deadline_loss_rate", &ns3::TvtAiEnv::deadline_loss_rate)
        .def_readwrite("delay_valid", &ns3::TvtAiEnv::delay_valid)
        .def_readwrite("deadline_loss_valid",
                       &ns3::TvtAiEnv::deadline_loss_valid)
        .def_readwrite("time_since_last_rx_ms",
                       &ns3::TvtAiEnv::time_since_last_rx_ms)
        .def_readwrite("logical_goodput_bps",
                       &ns3::TvtAiEnv::logical_goodput_bps)

        .def_readwrite("logical_tx_packets", &ns3::TvtAiEnv::logical_tx_packets)
        .def_readwrite("logical_rx_packets", &ns3::TvtAiEnv::logical_rx_packets)
        .def_readwrite("logical_deadline_success",
                       &ns3::TvtAiEnv::logical_deadline_success)
        .def_readwrite("logical_deadline_miss",
                       &ns3::TvtAiEnv::logical_deadline_miss)

        .def_readwrite("attempted_physical_copies_tx",
                       &ns3::TvtAiEnv::attempted_physical_copies_tx)
        .def_readwrite("successful_physical_copies_tx",
                       &ns3::TvtAiEnv::successful_physical_copies_tx)
        .def_readwrite("duplicate_copies_rx",
                       &ns3::TvtAiEnv::duplicate_copies_rx)
        .def_readwrite("duplicate_bytes_rx",
                       &ns3::TvtAiEnv::duplicate_bytes_rx)
        .def_readwrite("duplicated_logical_packets",
                       &ns3::TvtAiEnv::duplicated_logical_packets)
        .def_readwrite("window_logical_tx", &ns3::TvtAiEnv::window_logical_tx)
        .def_readwrite("window_attempted_copies",
                       &ns3::TvtAiEnv::window_attempted_copies)
        .def_readwrite("window_successful_copies",
                       &ns3::TvtAiEnv::window_successful_copies)
        .def_readwrite("window_duplicated_logical_packets",
                       &ns3::TvtAiEnv::window_duplicated_logical_packets)

        .def_readwrite("cohort_ready", &ns3::TvtAiEnv::cohort_ready)
        .def_readwrite("cohort_macro_id", &ns3::TvtAiEnv::cohort_macro_id)
        .def_readwrite("cohort_macro_tick", &ns3::TvtAiEnv::cohort_macro_tick)
        .def_readwrite("cohort_action", &ns3::TvtAiEnv::cohort_action)
        .def_readwrite("cohort_logical_tx", &ns3::TvtAiEnv::cohort_logical_tx)
        .def_readwrite("cohort_deadline_success",
                       &ns3::TvtAiEnv::cohort_deadline_success)
        .def_readwrite("cohort_deadline_miss",
                       &ns3::TvtAiEnv::cohort_deadline_miss)
        .def_readwrite("cohort_delay_sum_ms",
                       &ns3::TvtAiEnv::cohort_delay_sum_ms)
        .def_readwrite("cohort_delay_count", &ns3::TvtAiEnv::cohort_delay_count)
        .def_readwrite("cohort_attempted_copies",
                       &ns3::TvtAiEnv::cohort_attempted_copies)
        .def_readwrite("cohort_duplicated_logical_packets",
                       &ns3::TvtAiEnv::cohort_duplicated_logical_packets)
        .def_readwrite("cohort_uu_throughput_bps",
                       &ns3::TvtAiEnv::cohort_uu_throughput_bps)
        .def_readwrite("cohort_pc5_throughput_bps",
                       &ns3::TvtAiEnv::cohort_pc5_throughput_bps)
        .def_readwrite("cohort_logical_goodput_bps",
                       &ns3::TvtAiEnv::cohort_logical_goodput_bps)
        .def_readwrite("cohort_time_since_last_rx_ms",
                       &ns3::TvtAiEnv::cohort_time_since_last_rx_ms)
        .def_readwrite("cohort_backlog", &ns3::TvtAiEnv::cohort_backlog)

        .def_readwrite("formal_accounting_active",
                       &ns3::TvtAiEnv::formal_accounting_active)
        .def_readwrite("time_since_last_switch_ms",
                       &ns3::TvtAiEnv::time_since_last_switch_ms)
        .def_readwrite("is_switch", &ns3::TvtAiEnv::is_switch)
        .def_readwrite("done", &ns3::TvtAiEnv::done);

    py::class_<ns3::TvtAiAct>(m, "PyActStruct")
        .def(py::init<>())
        .def_readwrite("contract_version", &ns3::TvtAiAct::contract_version)
        .def_readwrite("decision_id", &ns3::TvtAiAct::decision_id)
        .def_readwrite("action", &ns3::TvtAiAct::action)
        .def_readwrite("macro_id", &ns3::TvtAiAct::macro_id)
        .def_readwrite("macro_tick", &ns3::TvtAiAct::macro_tick)
        .def_readwrite("infer_start_wall_s", &ns3::TvtAiAct::infer_start_wall_s)
        .def_readwrite("action_sent_wall_s", &ns3::TvtAiAct::action_sent_wall_s);

    using Interface = ns3::Ns3AiMsgInterfaceImpl<ns3::TvtAiEnv, ns3::TvtAiAct>;
    py::class_<Interface>(m, "Ns3AiMsgInterfaceImpl")
        .def(py::init<bool, bool, bool, uint32_t,
                      const char*, const char*, const char*, const char*>())
        .def("PyRecvBegin", &Interface::PyRecvBegin)
        .def("PyRecvEnd", &Interface::PyRecvEnd)
        .def("PySendBegin", &Interface::PySendBegin)
        .def("PySendEnd", &Interface::PySendEnd)
        .def("PyGetFinished", &Interface::PyGetFinished)
        .def("GetCpp2PyStruct", &Interface::GetCpp2PyStruct,
             py::return_value_policy::reference)
        .def("GetPy2CppStruct", &Interface::GetPy2CppStruct,
             py::return_value_policy::reference);
}
