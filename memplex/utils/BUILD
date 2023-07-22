package(
    default_visibility = [
        "//memplex:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "utils",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":memory",
    ],
)

py_library(
    name = "memory",
    srcs = ["memory.py"],
    srcs_version = "PY3",
)

py_test(
    name = "memory_test",
    size = "small",
    srcs = ["memory_test.py"],
    deps = [
        ":utils",
        "//memplex/models",
    ],
)
