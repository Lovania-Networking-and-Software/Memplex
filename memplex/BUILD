package_group(
    name = "friends",
    packages = ["//memplex/..."],
)

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

config_setting(
    name = "no_memplex_py_deps",
    define_values = {"no_memplex_py_deps": "true"},
    visibility = ["//visibility:public"],
)

py_library(
    name = "memplex",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":errors",
        "//memplex/layers",
        "//memplex/models",
        "//memplex/utils",
    ],
)

py_library(
    name = "errors",
    srcs = [
        "errors.py",
    ],
)
