#!/bin/sh
function stop_blazegraph {
  if [[ -z "$(lsof -t -i:9999)" ]]
  then
    echo "blazegraph isn't running..."
  else
    echo "stopping blazegraph..."
    kill $(lsof -t -i:9999) 2>/dev/null
  fi
}

function start_blazegraph {
  if [[ -z "$(lsof -t -i:9999)" ]]
  then
    echo "starting blazegraph..."
    if [[ "$2" == "wdbench" ]]
    then
    nohup java -server -Xmx4g -Dbigdata.propertyFile=blazegraph_wdbench.properties -jar blazegraph.jar &
    fi

    if [[ "$2" == "cordis" ]]
    then
      nohup java -server -Xmx4g -Dbigdata.propertyFile=blazegraph.properties -jar blazegraph.jar &
    fi
  else
    echo "blazegraph is already running..."
  fi
}

if [[ "$1" == "start" ]]
then
    start_blazegraph
fi

if [[ "$1" == "stop" ]]
then
    stop_blazegraph
fi
