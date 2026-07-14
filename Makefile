BIN := bin/gap
CMD := ./cmd/gap

.PHONY: all build test vet fmt clean

all: build

build:
	go build -o $(BIN) $(CMD)

test:
	go test ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

clean:
	rm -rf bin/
