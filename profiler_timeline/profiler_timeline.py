import tensorflow as tf
from tensorflow.profiler import Profiler
from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline


def test():
    shape = [2, 3]
    a = tf.get_variable(name="a", shape=shape, initializer=tf.random_normal_initializer(stddev=0.5))
    b = tf.get_variable(name="b", shape=shape, initializer=tf.random_normal_initializer(stddev=0.5))
    
    c = tf.multiply(a, b, name='c')
    d = tf.matmul(a, b, transpose_b=True, name='d')

    with tf.Session() as sess:
        profiler = Profiler(sess.graph)
        run_meta = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())
        sess.run([c, d], options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_meta)

        profiler.add_step(0, run_meta)
        profiler.profile_name_scope(options=option_builder.ProfileOptionBuilder.trainable_variables_parameter())
        opts = option_builder.ProfileOptionBuilder.time_and_memory()
        profiler.profile_operations(options=opts)

        opts = (option_builder.ProfileOptionBuilder(
            option_builder.ProfileOptionBuilder.time_and_memory()).with_step(0).with_timeline_output("test.out").build())
        profiler.profile_graph(options=opts)

        print("c: ", c.eval())
        print("d: ", d.eval())

        tl = timeline.Timeline(run_meta.step_stats)
        ctf = tl.generate_chrome_trace_format()
		
		# timeline.json = test.out
		
        with open('timeline.json', 'w') as f:
            f.write(ctf)
        # profiler.advise()
        

if __name__ == "__main__":
    test()

