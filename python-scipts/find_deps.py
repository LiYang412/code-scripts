#!/usr/bin/env python3

"""
  REFERENCE: https://github.com/google/iree/tree/main/build_tools/bazel_to_cmake
"""

import argparse
import os

all_deps = []

def setup_environment():
  """Sets up some environment globals."""
  global repo_root

  # Determine the repository root (two dir-levels up).
  repo_root = os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  print (repo_root)

def parse_arguments():
  global repo_root

  parser = argparse.ArgumentParser(
      description="Bazel to CMake conversion helper.")

  # Specify only one of these (defaults to --root_dir=iree).
  group = parser.add_mutually_exclusive_group()

  group.add_argument(
      "--root_dir",
      help="Converts the BUILD file in the given directory")

  args = parser.parse_args()

  return args

class Converter(object):
  """Conversion state tracking and full file template substitution."""

  def __init__(self):
    # Header appears after the license block but before `cmake_minimum_required(VERSION 3.10)`.
    self.header = ""
    # Body appears after `cmake_minimum_required(VERSION 3.10)`.
    self.body = ""

    self.first_error = None

  def convert(self):
    converted_content = (f"{self.header}\n\n"
                     	   f"cmake_minimum_required(VERSION 3.10)\n\n"
                         f"{self.body}")

    # Cleanup newline characters. This is more convenient than ensuring all
    # conversions are careful with where they insert newlines.
    converted_content = converted_content.replace("\n\n\n", "\n")
    converted_content = converted_content.rstrip() + "\n"

    return converted_content


def GetDict(obj):
  ret = {}
  for k in dir(obj):
    if not k.startswith("_"):
      ret[k] = getattr(obj, k)
  return ret



EXPLICIT_TARGET_MAPPING = {
    # apollo cmake
    #3rd-party
    "@com_github_gflags_gflags//:gflags": ["${GFLAGS_LIBRARIES}"],
    "@com_google_googletest//:gtest_main": ["${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${GMOCK_LIBRARIES} pthread"],
    "@com_google_googletest//:gtest": ['${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${GMOCK_LIBRARIES} pthread'],
    "@com_google_absl//absl/strings": ["${ABSL_STRINGS_LIBRARYIES}"],
    "@sqlite3" : ["${SQLITE_LIBRARIES}"],
    "@eigen": [], #只有h文件
    "@osqp": ["${OSQP_LIBRARIES}"],
    "@com_google_glog//:glog" : ["${GLOG_LIBRARIES}"],
    "@com_google_protobuf//:protobuf" : ["${PROTOBUF_LIBRARIES}"],
    "@fastrtps" : ["${FASTRTPS_LIBRARIES}"],
    "@uuid" : ["${UUID_LIBRARIES}"],
    "@fastcdr" : ["${FASTCDR_LIBRARIES}"],
    
    #install: sudo apt-get install libncurses5-dev
    "@ncurses5" : ["ncurses"],

    #install: sudo apt-get install -y libpoco-dev
    "@poco//:PocoFoundation" : ["PocoFoundation"],

    #cyber libraries
    "//cyber/common:file": ["cyber::file"],
    "//cyber/common:macros": ["cyber::common_macros"],
    # "//cyber/common:log": ["cyber::log"],
    "//cyber": ["cyber::cyber"],

    #缺少的
    #"//cyber/tools/cyber_recorder:recorder"
    #"//cyber/service_discovery:node_manager"


    #find_deps.py generator
	 "//cyber:init" : ['cyber::init'],
	 ":multi_value_warehouse" : ['cyber::multi_value_warehouse'],
	 "//cyber/node:writer_base" : ['cyber::writer_base'],
	 ":record_writer" : ['cyber::record_writer'],
	 ":record_message" : ['cyber::record_message'],
	 "//cyber/scheduler:processor" : ['cyber::processor'],
	 ":service_manager" : ['cyber::service_manager'],
	 ":renderable_message" : ['cyber::renderable_message'],
	 ":history" : ['cyber::history'],
	 "//cyber/proto:record_cc_proto" : ['cyber::record_cc_proto'],
	 ":py_message" : ['cyber::py_message'],
	 ":py_message_traits" : ['cyber::py_message_traits'],
	 ":header_builder" : ['cyber::header_builder'],
	 ":base" : ['cyber::base'],
	 ":graph" : ['cyber::graph'],
	 ":record_viewer" : ['cyber::record_viewer'],
	 "//cyber/parameter:parameter_client" : ['cyber::parameter_client'],
	 "@local_config_python//:python_lib" : ['cyber::python_lib'],
	 ":receiver" : ['cyber::receiver'],
	 ":underlay_message" : ['cyber::underlay_message'],
	 ":message_header" : ['cyber::message_header'],
	 ":record_base" : ['cyber::record_base'],
	 ":session" : ['cyber::session'],
	 ":data_fusion" : ['cyber::data_fusion'],
	 ":general_message" : ['cyber::general_message'],
	 ":blocker_manager" : ['cyber::blocker_manager'],
	 ":reader_base" : ['cyber::reader_base'],
	 "//cyber/common:global_data" : ['cyber::global_data'],
	 "//cyber/node:writer" : ['cyber::writer'],
	 "//cyber/scheduler:processor_context" : ['cyber::processor_context'],
	 "//cyber/proto:role_attributes_cc_proto" : ['cyber::role_attributes_cc_proto'],
	 ":time" : ['cyber::time'],
	 ":endpoint" : ['cyber::endpoint'],
	 ":block" : ['cyber::block'],
	 ":shm_conf" : ['cyber::shm_conf'],
	 ":dispatcher" : ['cyber::dispatcher'],
	 "//cyber/base:signal" : ['cyber::signal'],
	 "//cyber/time:rate" : ['cyber::rate'],
	 "//cyber:cyber_core" : ['cyber::cyber_core'],
	 "@ncurses5" : ['ncurses'],
	 "//cyber/common:types" : ['cyber::types'],
	 ":parameter_service_names" : ['cyber::parameter_service_names'],
	 ":hybrid_receiver" : ['cyber::hybrid_receiver'],
	 ":notifier_base" : ['cyber::notifier_base'],
	 ":identity" : ['cyber::identity'],
	 "//cyber/scheduler:scheduler_factory" : ['cyber::scheduler_factory'],
	 "//cyber/message:protobuf_traits" : ['cyber::protobuf_traits'],
	 "//cyber:binary" : ['cyber::binary'],
	 "//cyber/common:file" : ['cyber::file'],
	 "//cyber/croutine:swap" : ['cyber::swap'],
	 ":protobuf_traits" : ['cyber::protobuf_traits'],
	 "//cyber/common" : ['cyber::common'],
	 ":timer_task" : ['cyber::timer_task'],
	 "@fastrtps" : ['${FASTRTPS_LIBRARIES}'],
	 "role" : ['cyber::role'],
	 "//cyber/proto:classic_conf_cc_proto" : ['cyber::classic_conf_cc_proto'],
	 "//cyber/transport:qos_profile_conf" : ['cyber::qos_profile_conf'],
	 "//cyber/service_discovery:role" : ['cyber::role'],
	 "//cyber/croutine:routine_factory" : ['cyber::routine_factory'],
	 "//cyber/record:record_file_reader" : ['cyber::record_file_reader'],
	 "//cyber/common:macros" : ['cyber::common_macros'],
	 ":record_file_writer" : ['cyber::record_file_writer'],
	 ":record_file_base" : ['cyber::record_file_base'],
	 ":intra_dispatcher" : ['cyber::intra_dispatcher'],
	 ":task_manager" : ['cyber::task_manager'],
	 ":posix_segment" : ['cyber::posix_segment'],
	 ":play_task_buffer" : ['cyber::play_task_buffer'],
	 "//cyber/message:message_traits" : ['cyber::message_traits'],
	 "//cyber/proto:unit_test_cc_proto" : ['cyber::unit_test_cc_proto'],
	 "//cyber/base:reentrant_rw_lock" : ['cyber::reentrant_rw_lock'],
	 "//cyber/logger:logger_util" : ['cyber::logger_util'],
	 ":timing_wheel" : ['cyber::timing_wheel'],
	 "//cyber/common:environment" : ['cyber::environment'],
	 ":common_component_example_lib" : ['cyber::common_component_example_lib'],
	 "//cyber/record:record_reader" : ['cyber::record_reader'],
	 ":listener_handler" : ['cyber::listener_handler'],
	 ":rtps_transmitter" : ['cyber::rtps_transmitter'],
	 ":cache_buffer" : ['cyber::cache_buffer'],
	 ":history_attributes" : ['cyber::history_attributes'],
	 "//cyber/time" : ['cyber::time'],
	 ":protobuf_factory" : ['cyber::protobuf_factory'],
	 ":data_notifier" : ['cyber::data_notifier'],
	 "//cyber/transport:shm_transmitter" : ['cyber::shm_transmitter'],
	 ":attributes_filler" : ['cyber::attributes_filler'],
	 "//cyber/tools/cyber_recorder/player" : ['cyber::player'],
	 "//cyber/scheduler:scheduler_classic" : ['cyber::scheduler_classic'],
	 ":rtps_dispatcher" : ['cyber::rtps_dispatcher'],
	 "//cyber/class_loader:class_loader_manager" : ['cyber::class_loader_manager'],
	 "@com_google_googletest//:gtest_main" : ['${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${GMOCK_LIBRARIES} pthread'],
	 ":multicast_notifier" : ['cyber::multicast_notifier'],
	 ":poll_data" : ['cyber::poll_data'],
	 "//cyber/scheduler" : ['cyber::scheduler'],
	 "//cyber/component" : ['cyber::component'],
	 ":transmitter" : ['cyber::transmitter'],
	 "//cyber" : ['cyber::cyber'],
	 ":play_param" : ['cyber::play_param'],
	 "//cyber/class_loader" : ['cyber::class_loader'],
	 ":node_service_impl" : ['cyber::node_service_impl'],
	 ":all_latest" : ['cyber::all_latest'],
	 "//cyber/transport:hybrid_transmitter" : ['cyber::hybrid_transmitter'],
	 ":data_visitor_base" : ['cyber::data_visitor_base'],
	 ":recoverer" : ['cyber::recoverer'],
	 "//cyber/data:data_visitor" : ['cyber::data_visitor'],
	 "//cyber/transport:intra_transmitter" : ['cyber::intra_transmitter'],
	 "//cyber/time:duration" : ['cyber::duration'],
	 "@local_config_python//:python_headers" : ['cyber::python_headers'],
	 "//cyber/base:unbounded_queue" : ['cyber::unbounded_queue'],
	 ":duration" : ['cyber::duration'],
	 "//cyber/proto:choreography_conf_cc_proto" : ['cyber::choreography_conf_cc_proto'],
	 ":warehouse_base" : ['cyber::warehouse_base'],
	 "//cyber/transport:attributes_filler" : ['cyber::attributes_filler'],
	 "//cyber/base:atomic_rw_lock" : ['cyber::atomic_rw_lock'],
	 ":spliter" : ['cyber::spliter'],
	 "//cyber/message:protobuf_factory" : ['cyber::protobuf_factory'],
	 ":blocker" : ['cyber::blocker'],
	 ":intra_transmitter" : ['cyber::intra_transmitter'],
	 ":section" : ['cyber::section'],
	 "//cyber/message:raw_message_traits" : ['cyber::raw_message_traits'],
	 "//cyber/base:atomic_hash_map" : ['cyber::atomic_hash_map'],
	 ":raw_message" : ['cyber::raw_message'],
	 "//cyber/transport:sub_listener" : ['cyber::sub_listener'],
	 "//cyber/proto:component_conf_cc_proto" : ['cyber::component_conf_cc_proto'],
	 "//cyber/base:bounded_queue" : ['cyber::bounded_queue'],
	 "//cyber/blocker:intra_reader" : ['cyber::intra_reader'],
	 "//cyber/blocker" : ['cyber::blocker'],
	 "//cyber/record:header_builder" : ['cyber::header_builder'],
	 "//cyber/record:record_writer" : ['cyber::record_writer'],
	 "//cyber/service_discovery:topology_manager" : ['cyber::topology_manager'],
	 ":play_task_producer" : ['cyber::play_task_producer'],
	 ":shm_dispatcher" : ['cyber::shm_dispatcher'],
	 ":environment" : ['cyber::environment'],
	 "//cyber/transport:identity" : ['cyber::identity'],
	 "//cyber/logger" : ['cyber::logger'],
	 "//cyber/common:util" : ['cyber::util'],
	 ":general_channel_message" : ['cyber::general_channel_message'],
	 "//cyber/examples/proto:examples_cc_proto" : ['cyber::examples_cc_proto'],
	 "//cyber/base:thread_pool" : ['cyber::thread_pool'],
	 ":client_base" : ['cyber::client_base'],
	 "//cyber/logger:async_logger" : ['cyber::async_logger'],
	 "@fastcdr" : ['${FASTCDR_LIBRARIES}'],
	 "@com_google_glog//:glog" : ['${GLOG_LIBRARIES}'],
	 ":writer" : ['cyber::writer'],
	 "//cyber/base:rw_lock_guard" : ['cyber::rw_lock_guard'],
	 "//cyber/message:py_message_traits" : ['cyber::py_message_traits'],
	 ":intra_receiver" : ['cyber::intra_receiver'],
	 ":single_value_warehouse" : ['cyber::single_value_warehouse'],
	 "//cyber/base:macros" : ['cyber::macros'],
	 ":readable_info" : ['cyber::readable_info'],
	 "//cyber/transport:underlay_message_type" : ['cyber::underlay_message_type'],
	 "//cyber/message:raw_message" : ['cyber::raw_message'],
	 ":channel_manager" : ['cyber::channel_manager'],
	 "//cyber/transport" : ['cyber::transport'],
	 "//cyber/blocker:intra_writer" : ['cyber::intra_writer'],
	 ":record_file_reader" : ['cyber::record_file_reader'],
	 "//cyber/class_loader/test:base" : ['cyber::base'],
	 "@com_google_protobuf//:protobuf" : ['${PROTOBUF_LIBRARIES}'],
	 "//cyber/base:object_pool" : ['cyber::object_pool'],
	 "//cyber/service" : ['cyber::service'],
	 "//cyber/base:thread_safe_queue" : ['cyber::thread_safe_queue'],
	 "//cyber/croutine:routine_context" : ['cyber::routine_context'],
	 "//cyber/common:log" : ['cyber::log'],
	 "//cyber/base:concurrent_object_pool" : ['cyber::concurrent_object_pool'],
	 "//cyber/record:record_message" : ['cyber::record_message'],
	 "//cyber/proto:proto_desc_cc_proto" : ['cyber::proto_desc_cc_proto'],
	 "//cyber/parameter" : ['cyber::parameter'],
	 ":condition_notifier" : ['cyber::condition_notifier'],
	 "//cyber/base:for_each" : ['cyber::for_each'],
	 ":underlay_message_type" : ['cyber::underlay_message_type'],
	 ":message_info" : ['cyber::message_info'],
	 "//cyber/proto:clock_cc_proto" : ['cyber::clock_cc_proto'],
	 "//cyber:state" : ['cyber::state'],
	 ":hybrid_transmitter" : ['cyber::hybrid_transmitter'],
	 ":service_base" : ['cyber::service_base'],
	 ":info" : ['cyber::info'],
	 "//cyber/scheduler:scheduler_choreography" : ['cyber::scheduler_choreography'],
	 "//cyber/proto:cyber_conf_cc_proto" : ['cyber::cyber_conf_cc_proto'],
	 ":class_loader" : ['cyber::class_loader'],
	 ":rtps_receiver" : ['cyber::rtps_receiver'],
	 "//cyber/scheduler:mutex_wrapper" : ['cyber::mutex_wrapper'],
	 ":py_timer" : ['cyber::py_timer'],
	 ":timer_component_example_lib" : ['cyber::timer_component_example_lib'],
	 "//cyber/croutine" : ['cyber::croutine'],
	 ":play_task" : ['cyber::play_task'],
	 "//cyber/node" : ['cyber::node'],
	 ":py_time" : ['cyber::py_time'],
	 ":py_cyber" : ['cyber::py_cyber'],
	 ":participant_listener" : ['cyber::participant_listener'],
	 ":channel_buffer" : ['cyber::channel_buffer'],
	 ":segment_factory" : ['cyber::segment_factory'],
	 "//cyber/scheduler:cv_wrapper" : ['cyber::cv_wrapper'],
	 ":py_parameter" : ['cyber::py_parameter'],
	 ":record_reader" : ['cyber::record_reader'],
	 "//cyber/event:perf_event_cache" : ['cyber::perf_event_cache'],
	 ":data_dispatcher" : ['cyber::data_dispatcher'],
	 "//cyber/component:timer_component" : ['cyber::timer_component'],
	 ":raw_message_traits" : ['cyber::raw_message_traits'],
	 "//cyber/scheduler:classic_context" : ['cyber::classic_context'],
	 "//cyber/service:client" : ['cyber::client'],
	 "//cyber/io" : ['cyber::io'],
	 ":reader" : ['cyber::reader'],
	 ":notifier_factory" : ['cyber::notifier_factory'],
	 "//cyber/base:wait_strategy" : ['cyber::wait_strategy'],
	 "//cyber/transport:history" : ['cyber::history'],
	 "//cyber/blocker:blocker_manager" : ['cyber::blocker_manager'],
	 "//cyber/proto:parameter_cc_proto" : ['cyber::parameter_cc_proto'],
	 ":node_manager" : ['cyber::node_manager'],
	 ":py_record" : ['cyber::py_record'],
	 "//cyber/transport:participant" : ['cyber::participant'],
	 ":participant" : ['cyber::participant'],
	 "//cyber/proto:run_mode_conf_cc_proto" : ['cyber::run_mode_conf_cc_proto'],
	 ":play_task_consumer" : ['cyber::play_task_consumer'],
	 "//cyber/sysmo" : ['cyber::sysmo'],
	 "//cyber/timer:timing_wheel" : ['cyber::timing_wheel'],
	 ":perf_event" : ['cyber::perf_event'],
	 ":cyber_topology_message" : ['cyber::cyber_topology_message'],
	 "//cyber/transport:underlay_message" : ['cyber::underlay_message'],
	 "@poco//:PocoFoundation" : ['PocoFoundation'],
	 ":segment" : ['cyber::segment'],
	 "//cyber/common:time_conversion" : ['cyber::time_conversion'],
	 "//cyber/timer" : ['cyber::timer'],
	 "//cyber/time:clock" : ['cyber::clock'],
	 ":timer_bucket" : ['cyber::timer_bucket'],
	 "//cyber/scheduler:pin_thread" : ['cyber::pin_thread'],
	 "//cyber/data" : ['cyber::data'],
	 "//cyber/record:record_viewer" : ['cyber::record_viewer'],
	 ":recorder" : ['cyber::recorder'],
	 "//cyber/logger:log_file_object" : ['cyber::log_file_object'],
	 ":poller" : ['cyber::poller'],
	 ":subscriber_listener" : ['cyber::subscriber_listener'],
	 ":clock" : ['cyber::clock'],
	 ":screen" : ['cyber::screen'],
	 "@uuid" : ['${UUID_LIBRARIES}'],
	 ":data_visitor" : ['cyber::data_visitor'],
	 "//cyber/proto:qos_profile_cc_proto" : ['cyber::qos_profile_cc_proto'],
	 "//cyber/proto:dag_conf_cc_proto" : ['cyber::dag_conf_cc_proto'],
	 ":manager" : ['cyber::manager'],
	 ":shm_transmitter" : ['cyber::shm_transmitter'],
	 "//cyber/scheduler:choreography_context" : ['cyber::choreography_context'],
	 "//cyber/base" : ['cyber::base'],
	 ":sub_listener" : ['cyber::sub_listener'],
	 ":qos_profile_conf" : ['cyber::qos_profile_conf'],
	 "//cyber/proto:topology_change_cc_proto" : ['cyber::topology_change_cc_proto'],
	 "@com_google_googletest//:gtest" : ['${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${GMOCK_LIBRARIES} pthread'],
	 ":shm_receiver" : ['cyber::shm_receiver'],
	 "//cyber/parameter:parameter_server" : ['cyber::parameter_server'],
	 "//cyber/task" : ['cyber::task'],
	 "//cyber/transport:rtps_transmitter" : ['cyber::rtps_transmitter'],
	 ":component_base" : ['cyber::component_base'],
	 "//cyber/message:py_message" : ['cyber::py_message'],
	 ":node_channel_impl" : ['cyber::node_channel_impl'],
	 ":general_message_base" : ['cyber::general_message_base'],
	 ":xsi_segment" : ['cyber::xsi_segment'],
	 ":parameter" : ['cyber::parameter'],
	 ":state" : ['cyber::state'],
	 ":poll_handler" : ['cyber::poll_handler'],
	 ":cyber_core" : ['cyber::cyber_core'],
	 ":writer_base" : ['cyber::writer_base'],
	 "//cyber/record" : ['cyber::record'],


    #apollo bazel bug
    "role": ["cyber::role"],

    #modules_common
    "//modules/common/configs/proto:vehicle_config_cc_proto": ["modules::vehicle_config_cc_proto"],
    "//modules/common/math:geometry": ["modules::geometry"],
    "//modules/common/proto:pnc_point_cc_proto": ["modules::pnc_point_cc_proto"],
    ":config_gflags": ["modules::config_gflags"],
    "vehicle_config_helper":["modules::vehicle_config_helper"],
    ":digital_filter_coefficients": ["modules::digital_filter_coefficients"],
    ":digital_filter": ["modules::digital_filter"],
    ":mean_filter": ["modules::mean_filter"],
    "//modules/common/util": ["modules::util"],
    "//modules/common/util:future": ["modules::future"],
    ":kv_db": ["modules::kv_db"],
    "//modules/common/adapters:adapter_gflags": ["modules::adapter_gflags"],
    "//modules/common/latency_recorder/proto:latency_record_cc_proto": ["modules::latency_record_cc_proto"],
    "//modules/common/util:message_util": ["modules::message_util"],
    ":angle": ["modules::angle"],
    ":cartesian_frenet_conversion": ["modules::cartesian_frenet_conversion"],
    ":curve_fitting": ["modules::curve_fitting"],
    ":euler_angles_zxy": ["modules::euler_angles_zxy"],
    ":factorial": ["modules::factorial"],
    ":geometry": ["modules::geometry"],
    ":integral": ["modules::integral"],
    ":kalman_filter": ["modules::kalman_filter"],
    ":linear_interpolation": ["modules::linear_interpolation"],
    ":lqr": ["modules::lqr"],
    ":mpc_osqp": ["modules::mpc_osqp"],
    ":quaternion": ["modules::quaternion"],
    ":search": ["modules::search"],
    ":sin_table": ["modules::sin_table"],
    ":vec2d": ["modules::vec2d"],
    # ":math_utils": ["modules::math_utils"],
    "//modules/common/util:string_util": ["modules::string_util"],
    "//modules/common/proto:geometry_cc_proto": ["modules::geometry_cc_proto"],
    ":matrix_operations": ["modules::matrix_operations"],
    "//modules/common/math:matrix_operations": ["modules::matrix_operations"],
    "//modules/common/math:linear_interpolation": ["modules::linear_interpolation"],
    "//modules/common/proto:pnc_point_cc_proto": ["modules::pnc_point_cc_proto"],
}
def _add_deps(list_name, targets):
  if targets is None:
    return ""
  for t in targets:
    all_deps.append(t)

class BuildFileFunctions(object):
  """Object passed to `exec` that has handlers for BUILD file functions."""

  def __init__(self, converter):
    self.converter = converter
    print ("build file function __init__")
    print (self.converter)

  def _convert_unimplemented_function(self, function, details=""):
    message = f"Unimplemented {function}: {details}"
    if not self.converter.first_error:
      self.converter.first_error = NotImplementedError(message)
    # Avoid submitting the raw results from non-strict runs. These are still
    # useful but are generally not safe to submit as-is. An upstream check
    # prevents changes with this phrase from being submitted.
    # Written as separate literals to avoid the check triggering here.
    submit_blocker = "DO" + " NOT" + " SUBMIT."
    self.converter.body += f"# {submit_blocker} {message}\n"

  # Functions with no mapping to CMake. Just ignore these.
  def load(self, *args, **kwargs):
    pass

  def cc_proto_library(self, **kwargs):
    pass

  def proto_library(self, **kwargs):
    pass

  def py_proto_library(self, **kwargs):
    pass

  def package(self, **kwargs):
    pass

  def iree_build_test(self, **kwargs):
    pass

  def test_suite(self, **kwargs):
    pass

  def config_setting(self, **kwargs):
    pass

  def filegroup(self,
                 name,
                 srcs=None,
                 hdrs=None,
                 textual_hdrs=None,
                 copts=None,
                 defines=None,
                 data=None,
                 deps=None,
                 testonly=None,
                 linkopts=None,
                 **kwargs):
    if srcs:
      pass
    pass
  
  def enforce_glob(self, files, **kwargs):
    pass
    return files

  def glob(self, include, exclude=None, exclude_directories=1):
    pass

  def select(self, *args, **kwargs):
    pass

  def py_binary(self, *args, **kwargs):
    pass

  def cpplint(self, *args, **kwargs):
    pass

  def exports_files(self, *args, **kwargs):
    pass

  def cc_binary(self,
                 name,
                 hdrs=None,
                 textual_hdrs=None,
                 srcs=None,
                 copts=None,
                 defines=None,
                 data=None,
                 deps=None,
                 testonly=None,
                 linkopts=None,
                 **kwargs):
    print ("enter cc_binary")
    deps_block = _add_deps("DEPS", deps)

  def cc_library(self,
                 name,
                 hdrs=None,
                 textual_hdrs=None,
                 srcs=None,
                 copts=None,
                 defines=None,
                 data=None,
                 deps=None,
                 testonly=None,
                 linkopts=None,
                 **kwargs):
    print ("enter cc_library")
    deps_block = _add_deps("DEPS", deps)

  def cc_test(self,
              name,
              hdrs=None,
              srcs=None,
              copts=None,
              defines=None,
              data=None,
              deps=None,
              tags=None,
              **kwargs):
    print ("enter cc_library")
    deps_block = _add_deps("DEPS", deps)


  def py_library(self,
                 name,
                 hdrs=None,
                 textual_hdrs=None,
                 srcs=None,
                 copts=None,
                 defines=None,
                 data=None,
                 deps=None,
                 testonly=None,
                 linkopts=None,
                 **kwargs):
                 pass

  def py_test(self,
                 name,
                 hdrs=None,
                 textual_hdrs=None,
                 srcs=None,
                 copts=None,
                 defines=None,
                 data=None,
                 deps=None,
                 testonly=None,
                 linkopts=None,
                 **kwargs):
                 pass

def convert_build_file(build_file_code, allow_partial_conversion=False):
  converter = Converter()
  exec(build_file_code, GetDict(BuildFileFunctions(converter)))
  converted_text = converter.convert()
  # print ("convert_build_file")
  # print (converted_text)
  # if not allow_partial_conversion and converter.first_error:
  #   raise converter.first_error  # pylint: disable=raising-bad-type
  return converted_text

def main(args):
  print ("enter main function")
  if args.root_dir:
    #   print (args.root_dir)
      root_directory_path = os.path.join(repo_root, args.root_dir)
    #   print(root_directory_path)
      for root, _, _ in os.walk(root_directory_path):
        # print (root)
        build_file_path = os.path.join(root, "BUILD")
        cmakelists_file_path = os.path.join(root, "CMakeLists.txt")
        
        #如果没有BUILD文件则continue，防止下面找不到open是一个空文件报错
        if not os.path.isfile(build_file_path):
            continue
        
        with open(build_file_path, "rt") as build_file:
            build_file_code = compile(build_file.read(), build_file_path, "exec")
            # print ("build_file_code:%s" % str(build_file_code))
            converted_build_file = convert_build_file(build_file_code, False)
            # print (converted_build_file)

        # print ("converted_build_file:\n\n%s\n" % str(converted_build_file))
        # with open(cmakelists_file_path, "wt") as cmakelists_file:
        #     cmakelists_file.write(converted_build_file)

def convert_deps_target(target):
  if target in EXPLICIT_TARGET_MAPPING:
    return EXPLICIT_TARGET_MAPPING[target]
  else:
      if ":" in target:
        target_name = target.rsplit(":")[-1]
        print (target.rsplit(":")[0])
        return (f"['cyber::{target_name}']")
      else:
        print (target)
        target_name = target.rsplit("/")[-1]
        print (target.rsplit("/")[1])
        return (f"['cyber::{target_name}']")

def deal_all_deps(all_deps):
    if all_deps is None:
      return ""
      # Flatten lists
    # all_deps = list(itertools.chain.from_iterable(all_deps))
    # # Remove duplicates
    all_deps = set(all_deps)
    # # Remove Falsey (None and empty string) values
    all_deps = filter(None, all_deps)
    root_directory_path = os.path.join(repo_root, args.root_dir)
    deps_file_path = os.path.join(root_directory_path, "deps.txt")
    with open(deps_file_path, "wt") as deps_file:
      for target in all_deps:
        target_deps_name = convert_deps_target(target)
        test =               (f' "{target}" : '
                              f"{target_deps_name}"
                              f",\n")
        print (test)
        deps_file.write(test)



if __name__ == "__main__":
    
    #set enviroment 
    setup_environment()

    #set args
    args=parse_arguments()
    if args.root_dir is None:
      print ("Run 'python3.6.x find_deps.py --help' for more information. ")
    else: 
      #main function
      main(args)
      
      #print (all_deps)

      #analyzer all deps
      deal_all_deps(all_deps)
