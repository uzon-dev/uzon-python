# uzon

A Python parser and generator for the [UZON](https://uzon.dev) typed data expression format — spec v0.5.

```python
import uzon

data = uzon.loads('''
    server is {
        host is "localhost"
        port is 8080 as u16
        debug is true
    }
''')
print(data["server"]["host"])   # "localhost"
print(data["server"]["port"])   # UzonInt(8080, 'u16')
```

```python
>>> uzon.__version__
'0.5.0'
```

## Installation

```
pip install uzon
```

Requires Python 3.10+. No dependencies.

## Quick Start

```python
import uzon

# Parse a UZON string
data = uzon.loads('name is "Alice", age is 30')

# Parse a .uzon file
data = uzon.load("config.uzon")

# Generate UZON text from a dict
text = uzon.dumps({"name": "Alice", "age": 30})

# Write UZON to a file
uzon.dump({"name": "Alice"}, "output.uzon")

# Plain mode — strip all type wrappers
plain = uzon.loads('port is 8080 as u16', plain=True)
assert type(plain["port"]) is int  # plain int, not UzonInt

# Deep merge for config overlays
base = uzon.loads('host is "localhost", port is 8080')
override = uzon.loads('port is 9090, debug is true')
merged = uzon.merge(base, override)
# {"host": "localhost", "port": 9090, "debug": True}

# JSON serialization
import json
data = uzon.loads('color is Red | Green | Blue')
json.dumps(data, default=uzon.json_default)

# Create typed values from Python
port = uzon.val.u16(8080)
score = uzon.val.f32(9.5)
```

## UZON Syntax Preview

```
// Scalars
name is "hello"
count is 42
ratio is 3.14
active is true
missing is null

// Typed numbers
port is 8080 as u16
temperature is -40 as i8
weight is 72.5 as f32

// Structs
server is {
    host is "0.0.0.0"
    port is 443 as u16
}

// Lists
tags are ["web", "api", "v2"]
matrix is [[1, 2], [3, 4]]

// Tuples
point is (10, 20)

// Enums
color is Red | Green | Blue

// Tagged unions
result is Ok("success") | Err("failed")

// Functions
double is fn(x: i32) -> i32 { x * 2 }

// Expressions
total is price * quantity
greeting is "Hello, " + name

// Copy-and-update
dev_server is server with { port is 3000 as u16 }

// Extension
extended is server extends { timeout is 30 }

// File imports
db is struct "database.uzon"
```

See the full [UZON specification](https://uzon.dev) for details.

---

## API Reference

### Core Functions

#### `uzon.loads(text, *, plain=False)`

Parse and evaluate a UZON string.

```python
def loads(text: str, *, plain: bool = False) -> dict[str, Any]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | UZON source text. |
| `plain` | `bool` | `False` | If `True`, recursively strip all type wrappers to plain Python equivalents. |

**Returns:** `dict[str, Any]` — A dict of evaluated bindings. Keys are binding names, values are evaluated UZON values.

**Raises:**
- `UzonSyntaxError` — Invalid UZON syntax.
- `UzonTypeError` — Type annotation or compatibility error.
- `UzonRuntimeError` — Evaluation error (e.g. undefined variable, overflow).
- `UzonCircularError` — Circular dependency between bindings.

**Examples:**

```python
# Basic parsing
data = uzon.loads('x is 10, y is "hello"')
assert data == {"x": 10, "y": "hello"}

# Typed values are preserved
data = uzon.loads('port is 8080 as u16')
assert isinstance(data["port"], uzon.UzonInt)
assert data["port"].type_name == "u16"
assert data["port"] == 8080  # compares as int

# Plain mode strips type wrappers
data = uzon.loads('port is 8080 as u16', plain=True)
assert type(data["port"]) is int

# Expressions are evaluated
data = uzon.loads('a is 10, b is a * 2')
assert data["b"] == 20
```

---

#### `uzon.dumps(value, *, indent=4)`

Generate UZON text from a Python dict.

```python
def dumps(value: dict[str, Any], *, indent: int = 4) -> str
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `value` | `dict[str, Any]` | *(required)* | A dict representing a UZON document. |
| `indent` | `int` | `4` | Number of spaces per indentation level. |

**Returns:** `str` — UZON source text.

**Examples:**

```python
text = uzon.dumps({"name": "Alice", "age": 30})
# 'name is "Alice"\nage is 30'

# With typed values
text = uzon.dumps({"port": uzon.val.u16(8080)})
# 'port is 8080 as u16'

# Custom indentation
text = uzon.dumps({"server": {"host": "localhost"}}, indent=2)
```

---

#### `uzon.load(source, *, plain=False)`

Parse and evaluate a UZON file.

```python
def load(source: str | Path, *, plain: bool = False) -> dict[str, Any]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `source` | `str \| Path` | *(required)* | Path to a `.uzon` file. |
| `plain` | `bool` | `False` | If `True`, strip all type wrappers. |

**Returns:** `dict[str, Any]` — A dict of evaluated bindings.

**Raises:**
- `FileNotFoundError` — File does not exist.
- `UzonSyntaxError` — Invalid UZON syntax.
- `UzonTypeError` — Type annotation or compatibility error.
- `UzonRuntimeError` — Evaluation error.
- `UzonCircularError` — Circular dependency or circular import.

**Examples:**

```python
data = uzon.load("config.uzon")
data = uzon.load(Path("config.uzon"), plain=True)
```

---

#### `uzon.dump(value, dest, *, indent=4)`

Write UZON text to a file.

```python
def dump(value: dict[str, Any], dest: str | Path, *, indent: int = 4) -> None
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `value` | `dict[str, Any]` | *(required)* | A dict representing a UZON document. |
| `dest` | `str \| Path` | *(required)* | Path to write to. |
| `indent` | `int` | `4` | Number of spaces per indentation level. |

**Returns:** `None`

**Examples:**

```python
uzon.dump({"name": "Alice"}, "output.uzon")
uzon.dump(data, Path("config.uzon"), indent=2)
```

---

#### `uzon.merge(base, override)`

Deep-merge two UZON structs (dicts), returning a new dict.

```python
def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]
```

Nested dicts are merged recursively. All other values in `override` replace `base`. Type metadata (`UzonStruct.type_name`) is preserved from `override` if present, otherwise from `base`.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `base` | `dict[str, Any]` | *(required)* | The base struct. |
| `override` | `dict[str, Any]` | *(required)* | The override struct. Values here take precedence. |

**Returns:** `dict[str, Any]` — A new merged dict. Inputs are not mutated.

**Raises:**
- `TypeError` — If either argument is not a dict.

**Examples:**

```python
base = uzon.loads('host is "localhost", port is 8080')
override = uzon.loads('port is 9090, debug is true')
merged = uzon.merge(base, override)
# {"host": "localhost", "port": 9090, "debug": True}

# Nested merge
base = uzon.loads('db is { host is "localhost", port is 5432 }')
override = uzon.loads('db is { port is 3306 }')
merged = uzon.merge(base, override)
# {"db": {"host": "localhost", "port": 3306}}

# UzonStruct type_name is preserved
base = uzon.val.struct({"x": 1}, type_name="Point")
override = {"x": 2}
merged = uzon.merge(base, override)
assert isinstance(merged, uzon.UzonStruct)
assert merged.type_name == "Point"
```

---

#### `uzon.pretty_format(value, *, indent=2)`

Pretty-format a UZON value for debugging. Unlike `dumps()` which produces valid UZON source, `pretty_format()` produces a human-readable representation that shows type information.

```python
def pretty_format(value: Any, *, indent: int = 2) -> str
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `value` | `Any` | *(required)* | Any UZON value (dict, list, typed value, etc.). |
| `indent` | `int` | `2` | Number of spaces per indentation level. |

**Returns:** `str` — Multi-line human-readable representation.

**Examples:**

```python
data = uzon.loads('''
    server is {
        host is "localhost"
        port is 8080 as u16
    } called Server
''')
print(uzon.pretty_format(data))
# {
#   server:
#     { as Server
#       host: 'localhost'
#       port: 8080 as u16
#     }
# }
```

---

#### `uzon.json_default(obj)`

JSON serialization hook for UZON types. Pass as the `default` argument to `json.dumps()`.

```python
def json_default(obj: Any) -> Any
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `obj` | `Any` | *(required)* | The object that `json.dumps` could not serialize natively. |

**Returns:** A JSON-serializable Python value.

**Type mapping:**

| UZON Type | JSON Output |
|-----------|-------------|
| `UzonInt` / `UzonFloat` | number (handled natively by `json`) |
| `UzonEnum` | string (variant name) |
| `UzonUnion` | inner value (recursive) |
| `UzonTaggedUnion` | `{"_tag": str, "_value": ...}` |
| `UzonStruct` | object (handled natively — it's a `dict`) |
| `UzonUndefined` | `null` |
| `UzonFunction` | raises `TypeError` |

**Raises:**
- `TypeError` — If the object is a `UzonFunction` or otherwise not serializable.

**Examples:**

```python
import json

data = uzon.loads('''
    color is Red | Green | Blue
    result is Ok(42) | Err("fail")
    port is 8080 as u16
''')

text = json.dumps(data, default=uzon.json_default)
# {"color": "Red", "result": {"_tag": "Ok", "_value": 42}, "port": 8080}
```

---

### Value Factory (`uzon.val`)

The `val` object provides factory methods for creating typed UZON values from Python. Integer and float types are resolved dynamically — any bit width supported by the UZON spec works.

#### Integer Types: `val.i{N}(value)` / `val.u{N}(value)`

Create a typed integer. `i` = signed, `u` = unsigned. `N` = bit width.

```python
val.i8(value: int) -> UzonInt      # signed 8-bit:  -128 to 127
val.i16(value: int) -> UzonInt     # signed 16-bit: -32768 to 32767
val.i32(value: int) -> UzonInt     # signed 32-bit
val.i64(value: int) -> UzonInt     # signed 64-bit
val.i128(value: int) -> UzonInt    # signed 128-bit
val.u8(value: int) -> UzonInt      # unsigned 8-bit: 0 to 255
val.u16(value: int) -> UzonInt     # unsigned 16-bit: 0 to 65535
val.u32(value: int) -> UzonInt     # unsigned 32-bit
val.u64(value: int) -> UzonInt     # unsigned 64-bit
val.u128(value: int) -> UzonInt    # unsigned 128-bit
# ... any bit width: val.i256(), val.u1(), etc.
```

**Raises:**
- `TypeError` — If `value` is not an `int` (or is a `bool`).
- `OverflowError` — If `value` is outside the range for the type.

**Examples:**

```python
port = uzon.val.u16(8080)        # UzonInt(8080, 'u16')
big = uzon.val.i128(2**100)      # UzonInt(..., 'i128')
flag = uzon.val.u1(1)            # UzonInt(1, 'u1')

uzon.val.u8(256)                 # OverflowError: 256 out of u8 range [0, 255]
uzon.val.i8(-129)                # OverflowError: -129 out of i8 range [-128, 127]
uzon.val.u16("80")               # TypeError: val.u16() expects int, got str
```

#### Float Types: `val.f{N}(value)`

Create a typed float. Supported widths: 16, 32, 64, 80, 128.

```python
val.f16(value: float | int) -> UzonFloat
val.f32(value: float | int) -> UzonFloat
val.f64(value: float | int) -> UzonFloat
val.f80(value: float | int) -> UzonFloat
val.f128(value: float | int) -> UzonFloat
```

**Raises:**
- `TypeError` — If `value` is not a number (or is a `bool`).

**Examples:**

```python
score = uzon.val.f32(9.5)        # UzonFloat(9.5, 'f32')
ratio = uzon.val.f64(3.14)       # UzonFloat(3.14, 'f64')
half = uzon.val.f16(0.5)         # UzonFloat(0.5, 'f16')
uzon.val.f32(True)               # TypeError: val.f32() expects number, got bool
```

#### `val.struct(fields, *, type_name=None)`

Create a named UZON struct.

```python
def struct(fields: dict[str, Any], *, type_name: str | None = None) -> UzonStruct
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `fields` | `dict[str, Any]` | *(required)* | Dict of field names to values. |
| `type_name` | `str \| None` | `None` | Optional named type (e.g. `"Server"`). |

**Returns:** `UzonStruct` — A dict subclass with `type_name` metadata.

**Examples:**

```python
server = uzon.val.struct(
    {"host": "localhost", "port": uzon.val.u16(8080)},
    type_name="Server",
)
assert server.type_name == "Server"
assert server["host"] == "localhost"
```

#### `val.enum(value, variants, *, type_name=None)`

Create a UZON enum value.

```python
def enum(
    value: str,
    variants: list[str],
    *,
    type_name: str | None = None,
) -> UzonEnum
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `value` | `str` | *(required)* | The active variant name. |
| `variants` | `list[str]` | *(required)* | All allowed variant names. |
| `type_name` | `str \| None` | `None` | Optional named type (e.g. `"Color"`). |

**Raises:**
- `ValueError` — If `value` is not in `variants`.

**Examples:**

```python
color = uzon.val.enum("Red", ["Red", "Green", "Blue"], type_name="Color")
assert color.value == "Red"
assert color == "Red"  # compares with str
```

#### `val.union(value, types, *, type_name=None)`

Create a UZON untagged union value.

```python
def union(
    value: Any,
    types: list[str],
    *,
    type_name: str | None = None,
) -> UzonUnion
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `value` | `Any` | *(required)* | The inner value. |
| `types` | `list[str]` | *(required)* | List of allowed type names. |
| `type_name` | `str \| None` | `None` | Optional named type. |

**Examples:**

```python
flexible = uzon.val.union(42, ["string", "i32"])
assert flexible.value == 42
assert flexible == 42  # transparent comparison
```

#### `val.tagged(tag, value, variants, *, type_name=None)`

Create a UZON tagged union value.

```python
def tagged(
    tag: str,
    value: Any,
    variants: dict[str, str | None],
    *,
    type_name: str | None = None,
) -> UzonTaggedUnion
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tag` | `str` | *(required)* | The active variant tag. |
| `value` | `Any` | *(required)* | The inner value. |
| `variants` | `dict[str, str \| None]` | *(required)* | Map of variant names to their payload types (`None` for no payload). |
| `type_name` | `str \| None` | `None` | Optional named type (e.g. `"Result"`). |

**Raises:**
- `ValueError` — If `tag` is not in `variants`.

**Examples:**

```python
result = uzon.val.tagged(
    "Ok", "done",
    {"Ok": "string", "Err": "string"},
    type_name="Result",
)
assert result.tag == "Ok"
assert result.value == "done"
```

---

### Types

All type wrappers are importable from `uzon`:

```python
from uzon import (
    UzonInt, UzonFloat, UzonEnum, UzonUnion,
    UzonTaggedUnion, UzonStruct, UzonTypedList,
    UzonFunction, UzonUndefined,
)
```

#### `UzonInt`

```python
class UzonInt(int)
```

Typed UZON integer preserving width annotation. Subclasses `int` — usable anywhere a plain `int` is expected.

**Constructor:**

```python
UzonInt(value: int, type_name: str, *, adoptable: bool = False)
```

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `type_name` | `str` | Width annotation (e.g. `"i32"`, `"u16"`). |
| `adoptable` | `bool` | `True` if this is an untyped literal that can adopt another integer type. |

**Methods:**
- `to_plain() -> int` — Return the underlying plain `int`.

**String representations:**
- `repr(x)` → `UzonInt(10, 'i32')`
- `str(x)` → `10` (plain integer string)

**Arithmetic:** All arithmetic operators (`+`, `-`, `*`, `//`, `%`, `**`, `&`, `|`, `^`, `<<`, `>>`, unary `-`/`+`/`abs`/`~`, `round`) return `UzonInt` with the same `type_name`. If the result overflows the type's range, `OverflowError` is raised.

```python
x = uzon.val.i8(100)
y = x + 20                  # UzonInt(120, 'i8')
z = x + 30                  # OverflowError: Result 130 overflows i8 range [-128, 127]
assert isinstance(x + 1, uzon.UzonInt)
assert (x + 1).type_name == "i8"

# Direct construction (prefer val factory)
n = UzonInt(42, "i32")
```

---

#### `UzonFloat`

```python
class UzonFloat(float)
```

Typed UZON float preserving width annotation. Subclasses `float`.

**Constructor:**

```python
UzonFloat(value: float, type_name: str, *, adoptable: bool = False)
```

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `type_name` | `str` | Width annotation (e.g. `"f32"`, `"f64"`). |
| `adoptable` | `bool` | `True` if this is an untyped literal that can adopt another float type. |

**Methods:**
- `to_plain() -> float` — Return the underlying plain `float`.

**String representations:**
- `repr(x)` → `UzonFloat(1.5, 'f32')`
- `str(x)` → `1.5` (plain float string)

**Arithmetic:** Operators `+`, `-`, `*`, `/`, `//`, `%`, `**`, unary `-`/`+`/`abs`, `round` return `UzonFloat` with the same `type_name`. Bitwise operators (`&`, `|`, `^`, `<<`, `>>`, `~`) are not supported on floats.

```python
x = uzon.val.f32(1.5)
y = x * 2.0                 # UzonFloat(3.0, 'f32')
z = x / 3.0                 # UzonFloat(0.5, 'f32')
assert isinstance(y, uzon.UzonFloat)
```

---

#### `UzonEnum`

```python
class UzonEnum
```

Enum value — one variant from a defined set.

**Constructor:**

```python
UzonEnum(value: str, variants: list[str], type_name: str | None = None)
```

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `value` | `str` | The active variant name. |
| `variants` | `list[str]` | All allowed variant names. |
| `type_name` | `str \| None` | Named type, if declared with `called`. |

**Methods:**
- `to_plain() -> str` — Return the variant name as a plain string.

**String representations:**
- `repr(x)` → `UzonEnum('Red', type=Color)`
- `str(x)` → `Red` (variant name)

**Equality:** Compares equal to other `UzonEnum` with the same `value` and `type_name`, or to a plain `str` matching `value`. Hashable.

**Pattern matching (Python 3.10+):** `__match_args__ = ("value",)`

```python
data = uzon.loads('color is Red | Green | Blue')
color = data["color"]
assert color == "Red"        # compare with str
assert color.value == "Red"
assert color.variants == ["Red", "Green", "Blue"]
print(f"Color: {color}")     # "Color: Red"

match color:
    case UzonEnum("Red"):
        print("got red")
```

---

#### `UzonUnion`

```python
class UzonUnion
```

Untagged union — a value with one of several possible types.

**Constructor:**

```python
UzonUnion(value: Any, types: list[str], type_name: str | None = None)
```

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `value` | `Any` | The inner value. |
| `types` | `list[str]` | Allowed type names. |
| `type_name` | `str \| None` | Named type, if declared with `called`. |

**Methods:**
- `to_plain() -> Any` — Return the inner value.

**Transparent access:** `[]`, `len()`, `iter()`, `in`, and `bool()` delegate to the inner value.

**Equality:** Compares inner values — `UzonUnion(42) == 42` is `True`. Hashable (delegates to inner value).

**Pattern matching (Python 3.10+):** `__match_args__ = ("value",)`

```python
data = uzon.loads('value is 42 as i32 | string')
v = data["value"]
assert v == 42
assert v.types == ["i32", "string"]

match v:
    case UzonUnion(int(n)):
        print(f"integer: {n}")
```

---

#### `UzonTaggedUnion`

```python
class UzonTaggedUnion
```

Tagged union — a value paired with an explicit variant tag.

**Constructor:**

```python
UzonTaggedUnion(value: Any, tag: str, variants: dict[str, str | None], type_name: str | None = None)
```

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `value` | `Any` | The inner value (payload). |
| `tag` | `str` | The active variant tag. |
| `variants` | `dict[str, str \| None]` | Map of variant names to payload types. |
| `type_name` | `str \| None` | Named type, if declared with `called`. |

**Methods:**
- `to_plain() -> Any` — Return the inner value.

**String representations:**
- `repr(x)` → `UzonTaggedUnion(42, tag='Ok', type=Result)`
- `str(x)` → `42` (inner value's string)

**Transparent access:** `[]`, `len()`, `iter()`, `in` delegate to the inner value.

**Equality:** Compares both `tag` AND inner `value`. Hashable.

**Pattern matching (Python 3.10+):** `__match_args__ = ("tag", "value")`

```python
data = uzon.loads('result is Ok(42) | Err("fail")')
match data["result"]:
    case UzonTaggedUnion(tag="Ok", value=v):
        print(f"Success: {v}")
    case UzonTaggedUnion(tag="Err", value=e):
        print(f"Error: {e}")
```

---

#### `UzonStruct`

```python
class UzonStruct(dict)
```

Dict subclass that preserves a named type annotation for round-trip fidelity. Used when a struct has an explicit `called TypeName` annotation.

**Constructor:**

```python
UzonStruct(mapping: dict | None = None, type_name: str | None = None)
```

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `type_name` | `str \| None` | The named type, or `None` for anonymous structs. |

All standard `dict` operations work — `[]`, `.get()`, `.keys()`, `.values()`, `.items()`, `in`, `len()`, iteration, etc.

```python
data = uzon.loads('''
    server is {
        host is "localhost"
        port is 8080
    } called Server
''')
server = data["server"]
assert isinstance(server, uzon.UzonStruct)
assert server.type_name == "Server"
assert server["host"] == "localhost"
```

---

#### `UzonTypedList`

```python
class UzonTypedList(list)
```

List subclass that preserves element type annotation for round-trip fidelity. Produced when a list has an explicit `as [Type]` annotation.

**Constructor:**

```python
UzonTypedList(elements: list, element_type: str | None = None)
```

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `element_type` | `str \| None` | Element type annotation (e.g. `"i32"`, `"Server"`). |

All standard `list` operations work. Subclass of `list`.

```python
data = uzon.loads('scores is [95, 87, 72] as [i32]')
scores = data["scores"]
assert isinstance(scores, uzon.UzonTypedList)
assert scores.element_type == "i32"
assert scores[0] == 95
```

---

#### `UzonFunction`

```python
class UzonFunction
```

Function value — a closure capturing its definition scope. Produced when parsing `fn(params) -> ReturnType { body }` expressions.

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `params` | `list[tuple[str, str, Any]]` | Parameter list: `[(name, type_name, default_or_None), ...]`. |
| `return_type` | `str` | Return type name. |
| `type_name` | `str \| None` | Named type, if declared with `called`. |

**Methods:**
- `signature() -> tuple[tuple[str, ...], str]` — Return `(param_types, return_type)` for structural comparison.

Functions are not directly callable from Python — they are evaluated within UZON expressions.

```python
data = uzon.loads('double is fn(x: i32) -> i32 { x * 2 }, result is double(21)')
assert isinstance(data["double"], uzon.UzonFunction)
assert data["result"] == 42
```

> **Note:** `UzonFunction` and `UzonBuiltinFunction` are not JSON-serializable — `json_default()` raises `TypeError` for them.

---

#### `UzonUndefined`

```python
UzonUndefined: _UzonUndefinedType
```

Singleton sentinel representing UZON's undefined state — the absence of a value. Distinct from `None` (`null` in UZON).

- `bool(UzonUndefined)` is `False`.
- Safe for identity comparison: `value is UzonUndefined`.
- Copy-safe: `copy()` and `deepcopy()` return the same singleton.

```python
data = uzon.loads('x is undefined')
assert data["x"] is uzon.UzonUndefined
assert not data["x"]  # falsy
```

---

### Errors

All UZON errors inherit from `UzonError` and carry source location information.

```python
from uzon import (
    UzonSyntaxError, UzonTypeError,
    UzonRuntimeError, UzonCircularError,
)
```

#### `UzonError`

```python
class UzonError(Exception)
```

Base class for all UZON errors.

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `line` | `int \| None` | Source line number (1-based). |
| `col` | `int \| None` | Source column number (1-based). |
| `file` | `str \| None` | Source file path, or `None` for string input. |

Error messages automatically include location info: `"File config.uzon, Line 5, col 12: ..."`.

---

#### `UzonSyntaxError`

```python
class UzonSyntaxError(UzonError)
```

Lexer and parser errors — invalid tokens, unexpected structure, malformed literals.

```python
try:
    uzon.loads('x is {')
except uzon.UzonSyntaxError as e:
    print(e)       # "Line 1, col 7: Expected '}' ..."
    print(e.line)  # 1
```

---

#### `UzonTypeError`

```python
class UzonTypeError(UzonError)
```

Type annotation and compatibility errors — type mismatches in `as`, `with`, `extends`, or operations between incompatible typed values.

```python
try:
    uzon.loads('x is "hello" as i32')
except uzon.UzonTypeError as e:
    print(e)  # type mismatch
```

---

#### `UzonRuntimeError`

```python
class UzonRuntimeError(UzonError)
```

Evaluation errors — undefined variables, division by zero, integer overflow, import file not found.

```python
try:
    uzon.loads('x is y + 1')  # y is not defined
except uzon.UzonRuntimeError as e:
    print(e)
```

---

#### `UzonCircularError`

```python
class UzonCircularError(UzonError)
```

Circular dependency between bindings, or circular file imports.

```python
try:
    uzon.loads('a is b, b is a')
except uzon.UzonCircularError as e:
    print(e)
```

---

### Plain Mode

When `plain=True` is passed to `loads()` or `load()`, all type wrappers are recursively stripped:

| UZON Type | Plain Python Type |
|-----------|-------------------|
| `UzonInt` | `int` |
| `UzonFloat` | `float` |
| `UzonEnum` | `str` (variant name) |
| `UzonUnion` | inner value (unwrapped) |
| `UzonTaggedUnion` | inner value (unwrapped) |
| `UzonStruct` | `dict` |
| `bool`, `str`, `None` | unchanged |
| `list` | `list` (elements recursively stripped) |
| `tuple` | `tuple` (elements recursively stripped) |

```python
data = uzon.loads('''
    port is 8080 as u16
    color is Red | Green | Blue
    result is Ok(42) | Err("fail")
''', plain=True)

assert type(data["port"]) is int           # not UzonInt
assert type(data["color"]) is str           # not UzonEnum
assert data["result"] == 42                 # unwrapped from TaggedUnion
```

---

## Links

- [UZON Specification](https://uzon.dev)
- [GitHub Repository](https://github.com/uzon-dev/uzon-python)

## License

MIT
