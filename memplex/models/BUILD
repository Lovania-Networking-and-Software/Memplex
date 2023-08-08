package(
    default_visibility = [
        "//memplex:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "models",
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
    deps = [
        "//memplex/layers/labeler",
    ],
)

py_test(
    name = "labeler_test",
    size = "small",
    srcs = ["labeler_test.py"],
    deps = [
        ":models",
    ],
)
