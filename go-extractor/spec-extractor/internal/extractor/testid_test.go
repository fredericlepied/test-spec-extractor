package extractor

import (
	"testing"
)

func TestExtractTestIDFromDescription(t *testing.T) {
	tests := []struct {
		name string
		desc string
		want string
	}{
		{
			name: "bracket format",
			desc: "some test [test_id:12345]",
			want: "12345",
		},
		{
			name: "paren format",
			desc: "some test (test_id:67890)",
			want: "67890",
		},
		{
			name: "no test ID",
			desc: "some test without ID",
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractTestIDFromDescription(tt.desc)
			if got != tt.want {
				t.Errorf("extractTestIDFromDescription() = %v, want %v", got, tt.want)
			}
		})
	}
}
