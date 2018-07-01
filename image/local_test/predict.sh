#!/bin/bash

payload=${1:-test_image.png}
content=${2:-image/png}

curl --data-binary @${payload} -H "Content-Type: ${content}" -v http://localhost:8080/invocations
