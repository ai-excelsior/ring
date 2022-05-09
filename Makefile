VERSION := $(or ${VERSION},0.0.1)

build-docker-seq2seq:
	docker build -f ring/ts/seq2seq/Dockerfile_cpu -t code.unianalysis.com:5050/unianalysis/ring/seq2seq:$(VERSION) --network=host .

build-docker-seq2seq-gpu:
	docker build -f ring/ts/seq2seq/Dockerfile_gpu -t code.unianalysis.com:5050/unianalysis/ring/seq2seq-gpu:$(VERSION) --network=host .
	docker push code.unianalysis.com:5050/unianalysis/ring/seq2seq-gpu:$(VERSION)

build-docker-nbeats:
	docker build -f ring/ts/nbeats/Dockerfile_cpu -t code.unianalysis.com:5050/unianalysis/ring/nbeats:$(VERSION) --network=host .

build-docker-nbeats-gpu:
	docker build -f ring/ts/nbeats/Dockerfile_gpu -t code.unianalysis.com:5050/unianalysis/ring/nbeats-gpu:$(VERSION) --network=host .
	docker push code.unianalysis.com:5050/unianalysis/ring/nbeats-gpu:$(VERSION)

build-docker-encdecad:
	docker build -f ring/anomal/enc_dec_ad/Dockerfile_cpu -t code.unianalysis.com:5050/unianalysis/ring/encdecad:$(VERSION) --network=host .

build-docker-encdecad-gpu:
	docker build -f ring/anomal/enc_dec_ad/Dockerfile_gpu -t code.unianalysis.com:5050/unianalysis/ring/encdecad-gpu:$(VERSION) --network=host .
	docker push code.unianalysis.com:5050/unianalysis/ring/encdecad-gpu:$(VERSION)

build-docker-dagmm:
	docker build -f ring/anomal/dagmm/Dockerfile_cpu -t code.unianalysis.com:5050/unianalysis/ring/dagmm:$(VERSION) --network=host .

build-docker-dagmm-gpu:
	docker build -f ring/anomal/dagmm/Dockerfile_gpu -t code.unianalysis.com:5050/unianalysis/ring/dagmm-gpu:$(VERSION) --network=host .
	docker push code.unianalysis.com:5050/unianalysis/ring/dagmm-gpu:$(VERSION)

build-cuda:
	docker build -f dockerfiles/cuda11.3/py3.8/Dockerfile -t code.unianalysis.com:5050/unianalysis/ring/cuda11.3-py3.8 dockerfiles/cuda11.3/py3.8
	docker push code.unianalysis.com:5050/unianalysis/ring/cuda11.3-py3.8:latest