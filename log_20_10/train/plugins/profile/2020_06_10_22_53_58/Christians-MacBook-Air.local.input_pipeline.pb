	����HP�@����HP�@!����HP�@	���Is�?���Is�?!���Is�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$����HP�@� �rh��?A�Mb�N�@Y�I+��?*	     $�@2F
Iterator::Model�~j�t��?!��I��V@)����K�?1�-�
��U@:Preprocessing2S
Iterator::Model::ParallelMap{�G�z�?!0n�C�@){�G�z�?10n�C�@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�� �rh�?!�u���@)L7�A`�?1>�Tr^@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatL7�A`�?!>�Tr^@)+�����?1�;�o�1@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor���S㥛?!������?)���S㥛?1������?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mb`?!׌��ϑ�?)����Mb`?1׌��ϑ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	� �rh��?� �rh��?!� �rh��?      ��!       "      ��!       *      ��!       2	�Mb�N�@�Mb�N�@!�Mb�N�@:      ��!       B      ��!       J	�I+��?�I+��?!�I+��?R      ��!       Z	�I+��?�I+��?!�I+��?JCPU_ONLY