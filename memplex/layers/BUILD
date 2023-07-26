package(
    default_visibility = [
        "//memplex:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":labeler_layers",
        "//memplex:errors",
        "//memplex/layers/attention",
    ],
)

py_library(
    name = "labeler_layers",
    srcs = ["labeler.py"],
    srcs_version = "PY3",
)
