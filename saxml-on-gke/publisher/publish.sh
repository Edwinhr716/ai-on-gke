#!/bin/bash
#load the variables from the configMap
source /publish/config/publish.conf

 while true; do
    while inotifywait -e delete_self /publish/config/publish.conf; do
        curl --request POST \
        --header "Content-type: application/json" \
        -s localhost:8888/unpublish --data \
            "{
            \"model\": \"${model}\"
            }";

        source /publish/config/publish.conf
        sleep 15

        curl --request POST \
        --header "Content-type: application/json" \
        -s localhost:8888/publish --data \
            "{
            \"model\": \"${model}\",
            \"model_path\": \"${model_path}\",
            \"checkpoint\": \"${checkpoint}\",
            \"replicas\": 1
            }"
    done
done
