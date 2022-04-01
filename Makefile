VERSION := $(or ${VERSION},0.0.1)

build-docker-seq2seq:
	docker build -f ring/ts/seq2seq/Dockerfile_cpu -t code.unianalysis.com:5050/unianalysis/ring/seq2seq:$(VERSION) --network=host .
	docker push code.unianalysis.com:5050/unianalysis/ring/seq2seq:$(VERSION)

build-docker-seq2seq-gpu:
	docker build -f ring/ts/seq2seq/Dockerfile_gpu -t code.unianalysis.com:5050/unianalysis/ring/seq2seq-gpu:$(VERSION) --network=host .
	docker push code.unianalysis.com:5050/unianalysis/ring/seq2seq-gpu:$(VERSION)