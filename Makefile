VERSION := $(or ${VERSION},0.0.1)

build-docker-seq2seq:
	docker build -f ring/ts/seq2seq/Dockerfile_cpu -t seq2seq:$(VERSION) --network=host .

