	����D�@����D�@!����D�@	�HU�?�HU�?!�HU�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$����D�@J+�@A�/�@�@Y��Q��?*	     \�@2F
Iterator::ModelR���Q�?!}N���!U@)sh��|?�?1�5q�TT@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[2]::ConcatenateZd;�O�?!�¼d<�%@)y�&1��?1/g�Z�d%@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat#��~j��?!�5#�@){�G�z�?1�ܖ�B�@:Preprocessing2S
Iterator::Model::ParallelMap���x�&�?!O�Lј	@)���x�&�?1O�Lј	@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap��v���?!)h�5'@)y�&1��?1/g�Z�d�?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenate[1]::FromTensor�~j�t�h?!���M�V�?)�~j�t�h?1���M�V�?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice����Mb`?!��ghs�?)����Mb`?1��ghs�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����MbP?!��ghs�?)����MbP?1��ghs�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	J+�@J+�@!J+�@      ��!       "      ��!       *      ��!       2	�/�@�@�/�@�@!�/�@�@:      ��!       B      ��!       J	��Q��?��Q��?!��Q��?R      ��!       Z	��Q��?��Q��?!��Q��?JCPU_ONLY