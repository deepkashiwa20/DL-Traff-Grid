#!/bin/bash
chmod 777 line concatenate normalize
for type in "in" "out"
do
    base_path="../../../TaxiBJ/DMVST_flow/"

    graph_path=$base_path"graph_embed_"$type".txt"
    embed1_path=$base_path"graph_embed_1_"$type".line"
    embed2_path=$base_path"graph_embed_2_"$type".line"
    norm1_path=$base_path"graph_embed_1_norm_"$type".line"
    norm2_path=$base_path"graph_embed_2_norm_"$type".line"
    topo_path=$base_path"graph_embed_1and2_"$type".txt"
    
    ./line -train $graph_path -output $embed1_path -size 16 -order 1 -binary 1
    ./line -train $graph_path -output $embed2_path -size 16 -order 2 -binary 1
    ./normalize -input $embed1_path -output $norm1_path -binary 1
    ./normalize -input $embed2_path -output $norm2_path -binary 1
    ./concatenate -input1  $norm1_path -input2 $norm2_path -output $topo_path
done