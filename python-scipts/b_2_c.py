#!/usr/bin/env python3

import argparse
import os
import itertools

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
      help="Converts all BUILD files under a root directory (defaults to iree/)",
      default="iree")

  args = parser.parse_args()

  # --dir takes precedence over --root_dir.
  # They are mutually exclusive, but the default value is still set.
  if args.root_dir is None:
    args.root_dir = None

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


def _expand_cmake_var(var):
  return "${" + var + "}"


def _convert_string_arg_block(name, value, quote=True, namespace = False):
  #  NAME
  #    "value"
  if value is None:
    return ""
  if quote:
    # return f'  {name}\n    "{value}"\n'
    return f'  #{name}\n    "{value}"\n'
  else:
    # return f"  {name}\n    {value}\n"
    if namespace:
      return f"  #{name}\n    modules::{value}\n"
    else:
      return f"  #{name}\n    {value}\n"

def _convert_proto_string_arg_block(name, value, quote=True, namespace = False):
  #  NAME
  #    "value"
  if value is None:
    return ""
  if quote:
    # return f'  {name}\n    "{value}"\n'
    return f'  #{name}\n    "{value}"\n'
  else:
    # return f"  {name}\n    {value}\n"
    if namespace:
      value = value.rsplit("proto")[0]
      print("value")
      print(value)
      return f"  #{name}\n    modules::{value}cc_proto\n"
    else:
      return f"  #{name}\n    {value}\n"


def _convert_string_list_block(name, values, quote=True, sort=False):
  # Note this deliberately distinguishes between an empty list (argument
  # explicitly specified) and None (argument left as default).
  if values is None:
    return ""

  if sort:
    values = sorted(values)
  if quote:
    values_list = "\n".join([f'    {v}' for v in values])
    # values_list = "\n".join([f'    "{v}"' for v in values])

  else:
    values_list = "\n".join([f"    {v}" for v in values])

  print (values_list)
#   return f"  {name}\n{values_list}\n"
  return f"  #{name}\n{values_list}\n"


def _convert_option_block(option, option_value):
  if option_value:
    # Note: this is a truthiness check as well as an existence check, e.g.
    # Bazel `testonly = False` will be handled correctly by this condition.
    return f"  {option}\n"
  else:
    return ""


def _convert_translate_tool_block(translate_tool):
  if translate_tool is None:
    return ""
  # Bazel target name to cmake binary name
  # Bazel `//iree/custom:custom-translate` -> CMake `iree_custom_custom-translate`
  translate_tool = translate_tool.replace(
      "//iree", "iree")  # iree/custom:custom-translate
  translate_tool = translate_tool.replace(":",
                                          "_")  # iree/custom_custom-translate
  translate_tool = translate_tool.replace("/",
                                          "_")  # iree_custom_custom-translate
  return _convert_string_arg_block("TRANSLATE_TOOL",
                                   translate_tool,
                                   quote=False)


def _convert_srcs_block(srcs):
  if srcs is None:
    return ""
  generated_srcs = [src for src in srcs if src.startswith(":")]
  srcs = [src for src in srcs if src not in generated_srcs]
  sets = []
  if srcs:
    sets.append(_convert_string_list_block("SRCS", srcs, sort=True))
  if generated_srcs:
    sets.append(
        _convert_string_list_block("GENERATED_SRCS",
                                   [src[1:] for src in generated_srcs],
                                   sort=True))
  return "\n".join(sets)


def _convert_td_file_block(td_file):
  if td_file.startswith("//iree"):
    # Bazel `//iree/dir/td_file.td`
    # -> CMake `${IREE_ROOT_DIR}/iree/dir/td_file.td
    # Bazel `//iree/dir/IR:td_file.td`
    # -> CMake `${IREE_ROOT_DIR}/iree/dir/IR/td_file.td
    td_file = td_file.replace("//iree", "${IREE_ROOT_DIR}/iree")
    td_file = td_file.replace(":", "/")
  return _convert_string_arg_block("TD_FILE", td_file)


EXPLICIT_TARGET_MAPPING = {
    # apollo cmake
    #3rd-party
    "@com_github_gflags_gflags//:gflags": ["${GFLAGS_LIBRARIES}"],
    "@com_google_googletest//:gtest_main": ["${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${GMOCK_LIBRARIES} pthread"],
    "@com_google_absl//absl/strings": ["${ABSL_STRINGS_LIBRARYIES}"],
    "@sqlite3" : ["${SQLITE_LIBRARIES}"],
    "@eigen": [], #只有h文件
    "@osqp": ["${OSQP_LIBRARIES}"],

    #cyber libraries
    "//cyber/common:file": ["cyber::file"],
    "//cyber/common:macros": ["cyber::common_macros"],
    "//cyber/common:log": ["cyber::log"],
    "//cyber": ["cyber::cyber"],

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
    ":math_utils": ["modules::math_utils"],
    "//modules/common/util:string_util": ["modules::string_util"],
    "//modules/common/proto:geometry_cc_proto": ["modules::geometry_cc_proto"],
    ":matrix_operations": ["modules::matrix_operations"],
    "//modules/common/math:matrix_operations": ["modules::matrix_operations"],
    "//modules/common/math:linear_interpolation": ["modules::linear_interpolation"],
    "//modules/common/proto:pnc_point_cc_proto": ["modules::pnc_point_cc_proto"],




    # Internal utilities to emulate various binary/library options.
    "//build_tools:default_linkopts": [],

    # absl
    "@com_google_absl//absl/flags:flag": ["absl::flags"],
    "@com_google_absl//absl/flags:parse": ["absl::flags_parse"],
    # LLVM
    "@llvm-project//llvm:IPO": ["LLVMipo"],
    # MLIR
    "@llvm-project//mlir:AllPassesAndDialects": ["MLIRAllDialects"],
    "@llvm-project//mlir:AffineToStandardTransforms": ["MLIRAffineToStandard"],
    "@llvm-project//mlir:CFGTransforms": ["MLIRSCFToStandard"],
    "@llvm-project//mlir:DialectUtils": [""],
    "@llvm-project//mlir:ExecutionEngineUtils": ["MLIRExecutionEngine"],
    "@llvm-project//mlir:GPUDialect": ["MLIRGPU"],
    "@llvm-project//mlir:GPUTransforms": ["MLIRGPU"],
    "@llvm-project//mlir:LinalgInterfaces": ["MLIRLinalg"],
    "@llvm-project//mlir:LinalgOps": ["MLIRLinalg"],
    "@llvm-project//mlir:LLVMDialect": ["MLIRLLVMIR"],
    "@llvm-project//mlir:LLVMTransforms": ["MLIRStandardToLLVM"],
    "@llvm-project//mlir:MathDialect": ["MLIRMath"],
    "@llvm-project//mlir:MemRefDialect": ["MLIRMemRef"],
    "@llvm-project//mlir:SCFToGPUPass": ["MLIRSCFToGPU"],
    "@llvm-project//mlir:SCFDialect": ["MLIRSCF"],
    "@llvm-project//mlir:StandardOps": ["MLIRStandard"],
    "@llvm-project//mlir:ShapeTransforms": ["MLIRShapeOpsTransforms"],
    "@llvm-project//mlir:SideEffects": ["MLIRSideEffectInterfaces"],
    "@llvm-project//mlir:SPIRVDialect": ["MLIRSPIRV"],
    "@llvm-project//mlir:TosaDialect": ["MLIRTosa"],
    "@llvm-project//mlir:ToLLVMIRTranslation": ["MLIRTargetLLVMIRExport"],
    "@llvm-project//mlir:mlir_c_runner_utils": ["MLIRExecutionEngine"],
    "@llvm-project//mlir:mlir-translate": ["mlir-translate"],
    "@llvm-project//mlir:MlirTableGenMain": ["MLIRTableGen"],
    "@llvm-project//mlir:MlirOptLib": ["MLIROptLib"],
    "@llvm-project//mlir:VectorOps": ["MLIRVector"],
    "@llvm-project//mlir:TensorDialect": ["MLIRTensor"],
    "@llvm-project//mlir:NVVMDialect": ["MLIRNVVMIR"],
    "@llvm-project//mlir:ROCDLDialect": ["MLIRROCDLIR"],
    # Vulkan
    "@iree_vulkan_headers//:vulkan_headers": ["Vulkan::Headers"],
    # Cuda
    "@cuda//:cuda_headers": ["cuda_headers"],
    # The Bazel target maps to the IMPORTED target defined by FindVulkan().
    "@vulkan_sdk//:sdk": ["Vulkan::Vulkan"],
    # Misc single targets
    "@com_google_benchmark//:benchmark": ["benchmark"],
    "@com_github_dvidelabs_flatcc//:flatcc": ["flatcc"],
    "@com_github_dvidelabs_flatcc//:runtime": ["flatcc::runtime"],
    #"@com_google_googletest//:gtest": ["gmock", "gtest"],
    "@renderdoc_api//:renderdoc_app": ["renderdoc_api::renderdoc_app"],
    "@spirv_cross//:spirv_cross_lib": ["spirv-cross-msl"],
    "@cpuinfo": ["cpuinfo"],
    "@vulkan_memory_allocator//:impl_header_only": ["vulkan_memory_allocator"],
}

def _convert_absl_target(target):
  # Default to a pattern substitution approach.
  # Take "absl::" and append the name part of the full target identifier, e.g.
  #   "@com_google_absl//absl/types:optional" -> "absl::optional"
  #   "@com_google_absl//absl/types:span"     -> "absl::span"
  if ":" in target:
    target_name = target.rsplit(":")[-1]
  else:
    target_name = target.rsplit("/")[-1]
  return ["absl::" + target_name]


def _convert_mlir_target(target):
  # Default to a pattern substitution approach.
  # Take "MLIR" and append the name part of the full target identifier, e.g.
  #   "@llvm-project//mlir:IR"   -> "MLIRIR"
  #   "@llvm-project//mlir:Pass" -> "MLIRPass"
  return ["MLIR" + target.rsplit(":")[-1]]


def _convert_llvm_target(target):
  # Default to a pattern substitution approach.
  # Prepend "LLVM" to the Bazel target name.
  #   "@llvm-project//llvm:AsmParser" -> "LLVMAsmParser"
  #   "@llvm-project//llvm:Core" -> "LLVMCore"
  return ["LLVM" + target.rsplit(":")[-1]]

def convert_external_target(target):
  if target in EXPLICIT_TARGET_MAPPING:
    return EXPLICIT_TARGET_MAPPING[target]
  else:
      if ":" in target:
        target_name = target.rsplit(":")[-1]
        # print (target.rsplit(":")[0])
        # print (target_name)
        return ["modules::" + target_name]
      else:
        target_name = target.rsplit("/")[-1]
        # print (target_name)
        # print (target.rsplit("/")[1])
        return ["modules::" + target_name]

  raise KeyError(f"No conversion found for target '{target}'")

def convert_proto_external_target(target):
  if target in EXPLICIT_TARGET_MAPPING:
    return EXPLICIT_TARGET_MAPPING[target]
  else:
      if ":" in target:
        target_name = target.rsplit(":")[-1]
        # print (target.rsplit(":")[0])
        # print (target_name)
        target_name = target_name.rsplit("proto")[0]
        return ["modules::" + target_name + "cc_proto"]
      else:
        target_name = target.rsplit("/")[-1]
        # print (target_name)
        # print (target.rsplit("/")[1])
        target_name = target_name.rsplit("proto")[0]
        return ["modules::" + target_name + "cc_proto"]

  raise KeyError(f"No conversion found for target '{target}'")

def _convert_target(target):
  target = convert_external_target(target)
  
  return target

def _convert_proto_target(target):
  target = convert_proto_external_target(target)
  
  return target

def _convert_target_list_block(list_name, targets):
  if targets is None:
    return ""
  
  targets = [_convert_target(t) for t in targets]
  # Flatten lists
  targets = list(itertools.chain.from_iterable(targets))
  # Remove duplicates
  targets = set(targets)
  # Remove Falsey (None and empty string) values
  targets = filter(None, targets)
  return _convert_string_list_block(list_name, targets, sort=True, quote=False)

def _convert_proto_target_list_block(list_name, targets):
  if targets is None:
    return ""
  
  targets = [_convert_proto_target(t) for t in targets]
  # Flatten lists
  targets = list(itertools.chain.from_iterable(targets))
  # Remove duplicates
  targets = set(targets)
  # Remove Falsey (None and empty string) values
  targets = filter(None, targets)
  return _convert_string_list_block(list_name, targets, sort=True, quote=False)

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

  # def proto_library(self, **kwargs):
  #   pass
  def proto_library(self,
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
    name_block = _convert_proto_string_arg_block("NAME", name, quote=False, namespace=True)
    copts_block = _convert_string_list_block("COPTS", copts, sort=False)
    defines_block = _convert_string_list_block("DEFINES", defines)
    srcs_block = _convert_srcs_block(srcs)
    deps_block = _convert_proto_target_list_block("DEPS", deps)
    testonly_block = _convert_option_block("TESTONLY", testonly)

    self.converter.body += (f"set(\n"
                            f"    PROTOS\n"
                            f"{srcs_block}"
                            f")\n\n")

    self.converter.body += (
      f"PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${{PROTOS}})\n\n"
      f"include_directories(${{PROTOBUF_INCLUDE_DIRS}})\n\n"
    )
    self.converter.body += (f"add_library(\n"
                            f"{name_block}"
                            f"{srcs_block}"
                            f"    ${{PROTO_SRCS}}\n"
                            f"    ${{PROTO_HDRS}}\n"
                            f")\n\n")
    if deps_block :
        self.converter.body += (f"target_link_libraries(\n"
                                f"{name_block}"
                                f"{deps_block}"
                                f"    ${{PROTOBUF_LIBRARIES}}\n"
                                f")\n\n")

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
    if exclude_directories != 1:
      self._convert_unimplemented_function("glob", "with exclude_directories")
    if exclude is None:
      exclude = []

    glob_vars = []
    for pattern in include:
      if "**" in pattern:
        # bazel's glob has some specific restrictions about crossing package
        # boundaries. We have no uses of recursive globs. Rather than try to
        # emulate them or silently give different behavior, just error out.
        # See https://docs.bazel.build/versions/master/be/functions.html#glob
        raise NotImplementedError("Recursive globs not supported")
      # Bazel `*.mlir` glob -> CMake Variable `_GLOB_X_MLIR`
      var = "_GLOB_" + pattern.replace("*", "X").replace(".", "_").upper()
      glob_vars.append(var)
      self.converter.body += (
          f"file(GLOB {var} LIST_DIRECTORIES false"
          f" RELATIVE {_expand_cmake_var('CMAKE_CURRENT_SOURCE_DIR')}"
          f" CONFIGURE_DEPENDS {pattern})\n")
    for pattern in exclude:
      if "**" in pattern:
        raise NotImplementedError("Recursive globs not supported")
      exclude_var = ("_GLOB_" +
                     pattern.replace("*", "X").replace(".", "_").upper())
      self.converter.body += (
          f"file(GLOB {exclude_var} LIST_DIRECTORIES false"
          f" RELATIVE {_expand_cmake_var('CMAKE_CURRENT_SOURCE_DIR')}"
          f" CONFIGURE_DEPENDS {pattern})\n")
      for glob_var in glob_vars:
        self.converter.body += (
            f"list(REMOVE_ITEM {glob_var} {_expand_cmake_var(exclude_var)})\n")
    return [_expand_cmake_var(var) for var in glob_vars]

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
    # print (name)
    # print (srcs)
    name_block = _convert_string_arg_block("NAME", name, quote=False)
    copts_block = _convert_string_list_block("COPTS", copts, sort=False)
    defines_block = _convert_string_list_block("DEFINES", defines)
    srcs_block = _convert_srcs_block(srcs)
    # if data:
      # print (data)
    # data_block = _convert_target_list_block("DATA", data)
    deps_block = _convert_target_list_block("DEPS", deps)
    testonly_block = _convert_option_block("TESTONLY", testonly)

    self.converter.body += (f"add_executable(\n"
                            f"{name_block}"
                            f"{srcs_block}"
                            f")\n\n")
    if deps_block :
        self.converter.body += (f"target_link_libraries(\n"
                                f"{name_block}"
                                f"{deps_block}"
                                f")\n\n")


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
    if linkopts:
      self._convert_unimplemented_function("linkopts")
    name_block = _convert_string_arg_block("NAME", name, quote=False,namespace=True)
    hdrs_block = _convert_string_list_block("HDRS", hdrs, sort=True)
    textual_hdrs_block = _convert_string_list_block("TEXTUAL_HDRS",
                                                    textual_hdrs,
                                                    sort=True)
    srcs_block = _convert_srcs_block(srcs)
    copts_block = _convert_string_list_block("COPTS", copts, sort=False)
    defines_block = _convert_string_list_block("DEFINES", defines)
    # data_block = _convert_target_list_block("DATA", data)
    deps_block = _convert_target_list_block("DEPS", deps)
    testonly_block = _convert_option_block("TESTONLY", testonly)

    self.converter.body += (f"add_library(\n"
                            f"{name_block}"
                            f"{srcs_block}"
                            f")\n\n")
    if deps_block :
        self.converter.body += (f"target_link_libraries(\n"
                                f"{name_block}"
                                f"{deps_block}"
                                f")\n\n")


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
    name_block = _convert_string_arg_block("NAME", name, quote=False)
    hdrs_block = _convert_string_list_block("HDRS", hdrs, sort=True)
    srcs_block = _convert_srcs_block(srcs)
    copts_block = _convert_string_list_block("COPTS", copts, sort=False)
    defines_block = _convert_string_list_block("DEFINES", defines)
    # data_block = _convert_target_list_block("DATA", data)
    deps_block = _convert_target_list_block("DEPS", deps)
    labels_block = _convert_string_list_block("LABELS", tags)

    self.converter.body += (f"add_library(\n"
                            f"{name_block}"
                            f"{srcs_block}"
                            f")\n\n")
    if deps_block :
        self.converter.body += (f"target_link_libraries(\n"
                                f"{name_block}"
                                f"{deps_block}"
                                f")\n\n")

def convert_build_file(build_file_code, allow_partial_conversion=False):
  converter = Converter()
  exec(build_file_code, GetDict(BuildFileFunctions(converter)))
  converted_text = converter.convert()
  print ("convert_build_file")
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
        
        header = "\n".join(["#" * 80] + [
                l.ljust(79) + "#" for l in [
                    f"# {root}/BUILD convert build file to",
                    f"# {root}/CMakeLists.txt",
                    "#",
                    "# CMake-only content.",
                    "#",
                    f"# To disable autogeneration for this file entirely, delete this header.",
                ]
        ] + ["#" * 80])

        # print (header)
        #add header
        converted_build_file = header + converted_build_file

        # print ("converted_build_file:\n\n%s\n" % str(converted_build_file))
        with open(cmakelists_file_path, "wt") as cmakelists_file:
            cmakelists_file.write(converted_build_file)

if __name__ == "__main__":
    #set enviroment 
    setup_environment()

    #set args
    args=parse_arguments()

    #main function
    main(args)
