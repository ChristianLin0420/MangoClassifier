	V-����@V-����@!V-����@	�ua���?�ua���?!�ua���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$V-����@      �?A�E��~�@Y�C�l���?*	     t�@2F
Iterator::Model�Zd;�?!UkD:#U@)��C�l��?1&Y�WT@:Preprocessing2S
Iterator::Model::ParallelMapsh��|?�?!�T��'~@)sh��|?�?1�T��'~@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeaty�&1��?!��*�]A@)�MbX9�?1�,
p[e@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�&1��?!¾tr�D@)D�l����?1�^�k@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip��C�l��?!S��/�.@)�A`��"�?1u�#�&@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap㥛� ��?!kp���@)�I+��?1Sc��|@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceL7�A`�?!���:�?)L7�A`�?1���:�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensorL7�A`�?!���:�?)L7�A`�?1���:�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	      �?      �?!      �?      ��!       "      ��!       *      ��!       2	�E��~�@�E��~�@!�E��~�@:      ��!       B      ��!       J	�C�l���?�C�l���?!�C�l���?R      ��!       Z	�C�l���?�C�l���?!�C�l���?JCPU_ONLY