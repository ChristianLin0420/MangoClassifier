	ˡE�3(�@ˡE�3(�@!ˡE�3(�@	�x?��?�x?��?!�x?��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ˡE�3(�@�~j�t��?A�n��%�@Y�I+� @*	     ��@2F
Iterator::Model���K7�?!I�[&qU@)��K7��?1Dz9�5T@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat��"��~�?!8�=�r#@)�~j�t��?1��	a	"@:Preprocessing2S
Iterator::Model::ParallelMap�z�G�?!��c"B�@)�z�G�?1��c"B�@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap;�O��n�?!��?�@);�O��n�?1��?�@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t��?!��	a	�?)�~j�t��?1��	a	�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor���Q��?!I��K���?)���Q��?1I��K���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�~j�t��?�~j�t��?!�~j�t��?      ��!       "      ��!       *      ��!       2	�n��%�@�n��%�@!�n��%�@:      ��!       B      ��!       J	�I+� @�I+� @!�I+� @R      ��!       Z	�I+� @�I+� @!�I+� @JCPU_ONLY