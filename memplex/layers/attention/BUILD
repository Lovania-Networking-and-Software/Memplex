package(
    default_visibility = [
        "//memplex:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "attention_layers",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":attention",
        ":memory_attention",
    ],
)

py_library(
    name = "memory_attention",
    srcs = ["memory_attention.py"],
    srcs_version = "PY3",
    deps = [
        ":attention",
    ],
)

py_library(
    name = "attention",
    srcs = ["attention.py"],
    srcs_version = "PY3",
)

py_test(
    name = "memory_attention_test",
    size = "small",
    srcs = ["memory_attention_test.py"],
    deps = [
        ":memory_attention",
    ],
)
