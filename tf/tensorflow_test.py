import tensorflow as tf


def main():
    x1 = tf.constant([1,2,3,4])
    x2 = tf.constant([5,6,7,8])

    product = tf.multiply(x1, x2)
    print(product)




if __name__ == "__main__":
    main()