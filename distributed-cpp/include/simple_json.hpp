#pragma once

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <cstdint>
#include <stdexcept>
#include <charconv>

namespace simple_json {

enum class Type { Null, Boolean, Number, String, Array, Object };

class Value {
public:
    Type type = Type::Null;
    bool bool_val = false;
    double num_val = 0.0;
    int64_t int_val = 0;
    std::string str_val;
    std::vector<Value> arr_val;
    std::map<std::string, Value> obj_val;

    Value() : type(Type::Null) {}
    Value(bool b) : type(Type::Boolean), bool_val(b) {}
    Value(int i) : type(Type::Number), num_val(i), int_val(i) {}
    Value(int64_t i) : type(Type::Number), num_val(static_cast<double>(i)), int_val(i) {}
    Value(uint64_t u) : type(Type::Number), num_val(static_cast<double>(u)), int_val(static_cast<int64_t>(u)) {}
    Value(double d) : type(Type::Number), num_val(d), int_val(static_cast<int64_t>(d)) {}
    Value(const char* s) : type(Type::String), str_val(s ? s : "") {}
    Value(std::string s) : type(Type::String), str_val(std::move(s)) {}
    Value(std::vector<Value> arr) : type(Type::Array), arr_val(std::move(arr)) {}
    Value(std::map<std::string, Value> obj) : type(Type::Object), obj_val(std::move(obj)) {}

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Boolean; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    bool as_bool(bool def = false) const { return is_bool() ? bool_val : def; }
    int64_t as_int64(int64_t def = 0) const { return is_number() ? int_val : def; }
    uint64_t as_uint64(uint64_t def = 0) const { return is_number() ? static_cast<uint64_t>(int_val) : def; }
    double as_double(double def = 0.0) const { return is_number() ? num_val : def; }
    const std::string& as_string() const { return str_val; }
    std::string as_string_or(const std::string& def) const { return is_string() ? str_val : def; }

    bool contains(const std::string& key) const {
        if (!is_object()) return false;
        return obj_val.find(key) != obj_val.end();
    }

    const Value& operator[](const std::string& key) const {
        static Value null_val;
        if (!is_object()) return null_val;
        auto it = obj_val.find(key);
        return it != obj_val.end() ? it->second : null_val;
    }

    Value& operator[](const std::string& key) {
        if (type != Type::Object) {
            type = Type::Object;
            obj_val.clear();
        }
        return obj_val[key];
    }

    const Value& operator[](const char* key) const {
        return (*this)[std::string(key)];
    }

    Value& operator[](const char* key) {
        return (*this)[std::string(key)];
    }

    const Value& operator[](size_t index) const {
        static Value null_val;
        if (!is_array() || index >= arr_val.size()) return null_val;
        return arr_val[index];
    }

    Value& operator[](size_t index) {
        if (type != Type::Array) {
            type = Type::Array;
            arr_val.clear();
        }
        if (index >= arr_val.size()) {
            arr_val.resize(index + 1);
        }
        return arr_val[index];
    }

    const Value& operator[](int index) const {
        return (*this)[static_cast<size_t>(index)];
    }

    Value& operator[](int index) {
        return (*this)[static_cast<size_t>(index)];
    }

    void push_back(Value v) {
        if (type != Type::Array) {
            type = Type::Array;
            arr_val.clear();
        }
        arr_val.push_back(std::move(v));
    }

    std::string serialize() const {
        std::ostringstream oss;
        serialize_to(oss);
        return oss.str();
    }

private:
    void serialize_to(std::ostream& os) const {
        switch (type) {
            case Type::Null: os << "null"; break;
            case Type::Boolean: os << (bool_val ? "true" : "false"); break;
            case Type::Number:
                if (num_val == static_cast<double>(int_val)) {
                    os << int_val;
                } else {
                    os << num_val;
                }
                break;
            case Type::String: {
                os << '"';
                for (char c : str_val) {
                    if (c == '"') os << "\\\"";
                    else if (c == '\\') os << "\\\\";
                    else if (c == '\b') os << "\\b";
                    else if (c == '\f') os << "\\f";
                    else if (c == '\n') os << "\\n";
                    else if (c == '\r') os << "\\r";
                    else if (c == '\t') os << "\\t";
                    else os << c;
                }
                os << '"';
                break;
            }
            case Type::Array: {
                os << '[';
                for (size_t i = 0; i < arr_val.size(); ++i) {
                    if (i > 0) os << ",";
                    arr_val[i].serialize_to(os);
                }
                os << ']';
                break;
            }
            case Type::Object: {
                os << '{';
                bool first = true;
                for (const auto& [k, v] : obj_val) {
                    if (!first) os << ",";
                    first = false;
                    os << '"' << k << "\":";
                    v.serialize_to(os);
                }
                os << '}';
                break;
            }
        }
    }
};

class Parser {
    std::string_view src;
    size_t pos = 0;

    void skip_whitespace() {
        while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\n' || src[pos] == '\r')) {
            pos++;
        }
    }

    char peek() {
        skip_whitespace();
        return pos < src.size() ? src[pos] : '\0';
    }

    char get() {
        skip_whitespace();
        return pos < src.size() ? src[pos++] : '\0';
    }

public:
    explicit Parser(std::string_view s) : src(s) {}

    Value parse() {
        skip_whitespace();
        if (pos >= src.size()) return Value();
        char c = src[pos];

        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();

        return Value();
    }

private:
    Value parse_object() {
        get(); // consume '{'
        Value obj(std::map<std::string, Value>{});
        skip_whitespace();
        if (peek() == '}') {
            get();
            return obj;
        }

        while (pos < src.size()) {
            skip_whitespace();
            if (peek() != '"') break;
            Value key = parse_string();
            skip_whitespace();
            if (get() != ':') break;
            Value val = parse();
            obj.obj_val[key.str_val] = std::move(val);

            skip_whitespace();
            char next = get();
            if (next == '}') return obj;
            if (next != ',') break;
        }
        return obj;
    }

    Value parse_array() {
        get(); // consume '['
        Value arr(std::vector<Value>{});
        skip_whitespace();
        if (peek() == ']') {
            get();
            return arr;
        }

        while (pos < src.size()) {
            Value val = parse();
            arr.arr_val.push_back(std::move(val));
            skip_whitespace();
            char next = get();
            if (next == ']') return arr;
            if (next != ',') break;
        }
        return arr;
    }

    Value parse_string() {
        get(); // consume '"'
        std::string s;
        while (pos < src.size()) {
            char c = src[pos++];
            if (c == '"') return Value(std::move(s));
            if (c == '\\' && pos < src.size()) {
                char esc = src[pos++];
                switch (esc) {
                    case '"': s += '"'; break;
                    case '\\': s += '\\'; break;
                    case '/': s += '/'; break;
                    case 'b': s += '\b'; break;
                    case 'f': s += '\f'; break;
                    case 'n': s += '\n'; break;
                    case 'r': s += '\r'; break;
                    case 't': s += '\t'; break;
                    default: s += esc; break;
                }
            } else {
                s += c;
            }
        }
        return Value(std::move(s));
    }

    Value parse_number() {
        size_t start = pos;
        if (src[pos] == '-') pos++;
        while (pos < src.size() && (src[pos] >= '0' && src[pos] <= '9')) pos++;
        bool is_float = false;
        if (pos < src.size() && src[pos] == '.') {
            is_float = true;
            pos++;
            while (pos < src.size() && (src[pos] >= '0' && src[pos] <= '9')) pos++;
        }
        if (pos < src.size() && (src[pos] == 'e' || src[pos] == 'E')) {
            is_float = true;
            pos++;
            if (pos < src.size() && (src[pos] == '+' || src[pos] == '-')) pos++;
            while (pos < src.size() && (src[pos] >= '0' && src[pos] <= '9')) pos++;
        }

        std::string_view num_str = src.substr(start, pos - start);
        double d = 0.0;
        int64_t i = 0;
        try {
            if (is_float) {
                d = std::stod(std::string(num_str));
                return Value(d);
            } else {
                i = std::stoll(std::string(num_str));
                return Value(i);
            }
        } catch (...) {
            return Value(0);
        }
    }

    Value parse_bool() {
        if (src.substr(pos, 4) == "true") {
            pos += 4;
            return Value(true);
        }
        if (src.substr(pos, 5) == "false") {
            pos += 5;
            return Value(false);
        }
        return Value(false);
    }

    Value parse_null() {
        if (src.substr(pos, 4) == "null") {
            pos += 4;
        }
        return Value();
    }
};

inline Value parse(std::string_view s) {
    Parser p(s);
    return p.parse();
}

} // namespace simple_json
