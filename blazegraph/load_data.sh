#!/bin/sh

function load_wdbench {
  java -Xmx32g -cp blazegraph.jar com.bigdata.rdf.store.DataLoader -defaultGraph http://example.com/wdbench blazegraph_wdbench.properties /mnt/graph/repos/mt_rdf/data/wdbench/splitted_nt
}


function load_cordis {
  java -cp blazegraph.jar com.bigdata.rdf.store.DataLoader -namespace kb blazegraph.properties ../data/cordis_temp/cordis.ttl
}

if [[ "$1" == "wdbench" ]]
then
    load_wdbench
fi

if [[ "$1" == "cordis" ]]
then
    load_cordis
fi
