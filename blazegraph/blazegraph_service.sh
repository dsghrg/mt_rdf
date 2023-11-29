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
    nohup java -server -Xmx4g -Dbigdata.propertyFile=blazegraph.properties -jar blazegraph.jar &
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
