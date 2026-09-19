// Package jsonx writes JSON that JavaScript can read.
//
// One thing separates the two: Go refuses to marshal a NaN or an infinity, and
// JavaScript writes them as null. That difference matters here because a design
// with a broken symbol evaluates to NaN by design — the symbol failed, the
// error says so, and the numbers downstream are meaningless but still have to
// reach the editor so it can show the error beside them. Without this, a single
// bad expression makes the whole report unsendable and the editor goes blank
// instead of explaining itself.
package jsonx

import (
	"encoding/json"
	"math"
	"reflect"
)

// Marshal encodes a value, writing any non-finite number as null.
//
// The common path is exactly encoding/json: nothing is copied and nothing is
// walked. Only when the standard encoder refuses does this rebuild the value
// with the offending numbers replaced, which happens on a design that is
// already reporting an error.
func Marshal(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err == nil {
		return raw, nil
	}
	var unsupported *json.UnsupportedValueError
	if !asUnsupportedValue(err, &unsupported) {
		return nil, err
	}
	return json.Marshal(sanitise(reflect.ValueOf(v)))
}

func asUnsupportedValue(err error, target **json.UnsupportedValueError) bool {
	for err != nil {
		if e, ok := err.(*json.UnsupportedValueError); ok {
			*target = e
			return true
		}
		type unwrapper interface{ Unwrap() error }
		u, ok := err.(unwrapper)
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

// sanitise rebuilds a value as the plain shapes encoding/json accepts, with
// non-finite numbers replaced by nil.
//
// It honours json tags, because the rebuilt value has to marshal to the same
// field names the fast path would have produced. A type with its own
// MarshalJSON is left to it: if that method can produce a non-finite number it
// could not have been marshalled by any path.
func sanitise(v reflect.Value) any {
	if !v.IsValid() {
		return nil
	}
	switch v.Kind() {
	case reflect.Float32, reflect.Float64:
		f := v.Float()
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil
		}
		return f

	case reflect.Pointer, reflect.Interface:
		if v.IsNil() {
			return nil
		}
		return sanitise(v.Elem())

	case reflect.Slice:
		if v.IsNil() {
			return nil
		}
		// A byte slice is base64 in JSON; leave that to the encoder.
		if v.Type().Elem().Kind() == reflect.Uint8 {
			return v.Interface()
		}
		out := make([]any, v.Len())
		for i := range out {
			out[i] = sanitise(v.Index(i))
		}
		return out

	case reflect.Array:
		out := make([]any, v.Len())
		for i := range out {
			out[i] = sanitise(v.Index(i))
		}
		return out

	case reflect.Map:
		if v.IsNil() {
			return nil
		}
		out := make(map[string]any, v.Len())
		for _, key := range v.MapKeys() {
			name, ok := keyName(key)
			if !ok {
				// A key JSON cannot express; the fast path would have refused
				// it too, so there is nothing sensible to write.
				continue
			}
			out[name] = sanitise(v.MapIndex(key))
		}
		return out

	case reflect.Struct:
		if marshaller, ok := v.Interface().(json.Marshaler); ok {
			return marshaller
		}
		if v.CanAddr() {
			if marshaller, ok := v.Addr().Interface().(json.Marshaler); ok {
				return marshaller
			}
		}
		return sanitiseStruct(v)

	default:
		return v.Interface()
	}
}

func sanitiseStruct(v reflect.Value) map[string]any {
	out := map[string]any{}
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if field.PkgPath != "" && !field.Anonymous {
			continue // unexported
		}
		name, omitEmpty, skip := fieldName(field)
		if skip {
			continue
		}
		value := v.Field(i)
		if field.Anonymous && name == "" {
			// An embedded struct's fields are the outer struct's fields.
			for key, inner := range sanitiseStruct(indirect(value)) {
				out[key] = inner
			}
			continue
		}
		if omitEmpty && isEmpty(value) {
			continue
		}
		out[name] = sanitise(value)
	}
	return out
}

func indirect(v reflect.Value) reflect.Value {
	for v.Kind() == reflect.Pointer && !v.IsNil() {
		v = v.Elem()
	}
	return v
}

// fieldName reads a field's json tag the way encoding/json does.
func fieldName(field reflect.StructField) (name string, omitEmpty bool, skip bool) {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}
	name = field.Name
	if tag != "" {
		parts := splitTag(tag)
		if parts[0] != "" {
			name = parts[0]
		} else if field.Anonymous {
			name = ""
		}
		for _, opt := range parts[1:] {
			if opt == "omitempty" {
				omitEmpty = true
			}
		}
	} else if field.Anonymous {
		name = ""
	}
	return name, omitEmpty, false
}

func splitTag(tag string) []string {
	var out []string
	start := 0
	for i := 0; i < len(tag); i++ {
		if tag[i] == ',' {
			out = append(out, tag[start:i])
			start = i + 1
		}
	}
	return append(out, tag[start:])
}

func isEmpty(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Pointer:
		return v.IsNil()
	}
	return false
}

func keyName(key reflect.Value) (string, bool) {
	switch key.Kind() {
	case reflect.String:
		return key.String(), true
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		raw, err := json.Marshal(key.Interface())
		if err != nil {
			return "", false
		}
		// Marshalling an integer gives its digits; JSON object keys are strings.
		return string(raw), true
	}
	if marshaller, ok := key.Interface().(interface{ MarshalText() ([]byte, error) }); ok {
		text, err := marshaller.MarshalText()
		if err != nil {
			return "", false
		}
		return string(text), true
	}
	return "", false
}
