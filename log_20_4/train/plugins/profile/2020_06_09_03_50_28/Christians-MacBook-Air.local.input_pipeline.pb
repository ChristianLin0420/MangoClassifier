	o��j@�@o��j@�@!o��j@�@	"��r��?"��r��?!"��r��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$o��j@�@�z�G��?A-����=�@Y�|?5^� @*	     ڠ@2F
Iterator::Model�G�z��?!�����U@)%��C��?1dpaG�T@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�l�����?!S�/r@)!�rh���?1�#��|�@:Preprocessing2S
Iterator::Model::ParallelMap�V-�?!�M�6U
@)�V-�?1�M�6U
@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip����S�?!9;!�y ,@)����Mb�?1U&�}��@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate333333�?!�&[ �@)y�&1��?1�߭�@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�Q���?!��V�D��?)�Q���?1��V�D��?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��~j�t�?!�]��/�?)��~j�t�?1�]��/�?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap
ףp=
�?!�JЙ�@)���Q��?1���@�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�z�G��?�z�G��?!�z�G��?      ��!       "      ��!       *      ��!       2	-����=�@-����=�@!-����=�@:      ��!       B      ��!       J	�|?5^� @�|?5^� @!�|?5^� @R      ��!       Z	�|?5^� @�|?5^� @!�|?5^� @JCPU_ONLY