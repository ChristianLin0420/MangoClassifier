	�p=
��@�p=
��@!�p=
��@	�Ǚr�i�?�Ǚr�i�?!�Ǚr�i�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�p=
��@=
ףp=�?A���S3��@Y�MbX9�?*	     @�@2F
Iterator::Model5^�I�?!l����R@)F����x�?1u�E]TQ@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap��"��~�?!n,�Rab5@)��"��~�?1n,�Rab5@:Preprocessing2S
Iterator::Model::ParallelMap�~j�t��?!y?r���@)�~j�t��?1y?r���@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat��~j�t�?!�4_�g@)����Mb�?1�T�x?r
@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�~j�t�x?!y?r����?)�~j�t�x?1y?r����?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mb`?!�T�x?r�?)����Mb`?1�T�x?r�?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor����Mb`?!�T�x?r�?)����Mb`?1�T�x?r�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	=
ףp=�?=
ףp=�?!=
ףp=�?      ��!       "      ��!       *      ��!       2	���S3��@���S3��@!���S3��@:      ��!       B      ��!       J	�MbX9�?�MbX9�?!�MbX9�?R      ��!       Z	�MbX9�?�MbX9�?!�MbX9�?JCPU_ONLY