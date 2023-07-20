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
        ":labeler",
        "//memplex:errors",
    ],
)

py_library(
    name = "labeler",
    srcs = ["labeler.py"],
    srcs_version = "PY3",
)

py_test(
    name = "labeler_test",
    size = "small",
    srcs = ["labeler_test.py"],
    deps = [
        ":layers",
    ],
)
