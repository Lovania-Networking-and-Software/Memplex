package(
    default_visibility = [
        "//memplex:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "labeler",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":labeler_norm_wrapper",
        ":labeler_prelu_layer",
        ":labeler_space_creator_layer",
    ],
)

py_library(
    name = "labeler_norm_wrapper",
    srcs = ["norm.py"],
    srcs_version = "PY3",
)

py_library(
    name = "labeler_prelu_layer",
    srcs = ["prelu.py"],
    srcs_version = "PY3",
    deps = ["//memplex:errors"],
)

py_library(
    name = "labeler_space_creator_layer",
    srcs = ["space_creator.py"],
    srcs_version = "PY3",
    deps = ["//memplex:errors"],
)
