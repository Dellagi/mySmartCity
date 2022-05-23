#!/usr/bin/env bash

if [[ "$(docker images -q mysmartcity 2> /dev/null)" == "" ]]; then
	docker build -t mysmartcity .
fi

docker run -p 8080:8080 -p 8000:8000 -it mysmartcity
