#!/bin/bash

set -eu

script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/extract-spec-md.sh"
workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
cd "$workdir"

assert_missing_value() {
    local option=$1 output
    if output=$(bash "$script" --go-root . --output-dir results "$@" 2>error.log); then
        printf 'Expected failure for: %s\n' "$*" >&2
        exit 1
    fi
    if [[ -n "$output" || $(<error.log) != *"Missing path for $option"* ||
          -e results || -e spec-md || -e ./--verbose ]]; then
        printf 'Expected a missing path error before extraction for: %s\n' "$*" >&2
        exit 1
    fi
}

for option in -g --go-root -p --py-root -o --output-dir; do
    assert_missing_value "$option"
    assert_missing_value "$option" ""
    assert_missing_value "$option" --help
    assert_missing_value "$option" --verbose
    assert_missing_value "$option" --py-root
done

for option in -h --help; do
    output=$(bash "$script" "$option" 2>error.log)
    [[ "$output" == Usage:* && ! -s error.log && ! -e spec-md ]]
done
