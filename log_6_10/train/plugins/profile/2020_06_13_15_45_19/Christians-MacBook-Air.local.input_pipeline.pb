	}?5^�4�@}?5^�4�@!}?5^�4�@	��A�� �?��A�� �?!��A�� �?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$}?5^�4�@`��"��@A�&1D3�@Y7�A`��@*	     T�@2F
Iterator::ModelB`��"� @!D���CnR@)333333 @12�[���Q@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenate��/�$�?!a���z0@)��(\���?1o�5�&E0@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap#��~j��?!U�V�6@)-����?1����@:Preprocessing2S
Iterator::Model::ParallelMapˡE����?!C�H��@)ˡE����?1C�H��@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�x�&1�?!��!�F:@)�A`��"�?1��Yͫ�?:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�������?! ɦ����?)�I+��?1��@���?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�~j�t�x?!Hy/[7��?)�~j�t�x?1Hy/[7��?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenate[1]::FromTensor����Mbp?!��t�$��?)����Mbp?1��t�$��?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice����Mb`?!��t�$�?)����Mb`?1��t�$�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	`��"��@`��"��@!`��"��@      ��!       "      ��!       *      ��!       2	�&1D3�@�&1D3�@!�&1D3�@:      ��!       B      ��!       J	7�A`��@7�A`��@!7�A`��@R      ��!       Z	7�A`��@7�A`��@!7�A`��@JCPU_ONLY