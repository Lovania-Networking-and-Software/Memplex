package(
    default_visibility = [
        "//memplex:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "attention",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":memory_attention",
        "//memplex:errors",
    ],
)

py_library(
    name = "base_dense_attention",
    srcs = ["base_dense_attention.py"],
    srcs_version = "PY3",
)

py_library(
    name = "memory_attention",
    srcs = ["memory_attention.py"],
    srcs_version = "PY3",
    deps = [
        ":base_dense_attention",
    ],
)

py_test(
    name = "memory_attention_test",
    size = "small",
    srcs = ["memory_attention_test.py"],
    deps = [
        ":memory_attention",
    ],
)
