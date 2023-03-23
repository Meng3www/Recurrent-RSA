import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def uniform_vector(length):
	return np.ones((length))/length

def make_initial_prior(initial_image_prior,initial_rationality_prior,initial_speaker_prior):
	
	return np.log(np.multiply.outer(initial_image_prior,np.multiply.outer(initial_rationality_prior,initial_speaker_prior)))


if __name__ == '__main__':
    x = [4, 6, 8]
    e_x = np.exp(x - np.max(x))
    print(e_x / e_x.sum())

    ################################
    initial_image_prior = uniform_vector(2)
    initial_rationality_prior = uniform_vector(1)
    initial_speaker_prior = uniform_vector(1)
    print("np.multiply.outer(initial_rationality_prior,initial_speaker_prior)\n",
          np.multiply.outer(initial_rationality_prior, initial_speaker_prior))
    print(np.multiply.outer([1.], [1.]))
    print("np.multiply.outer(initial_image_prior,np.multiply.outer(initial_rationality_prior,initial_speaker_prior)) \n",
          np.multiply.outer(initial_image_prior,np.multiply.outer(initial_rationality_prior,initial_speaker_prior)))
    print(np.multiply.outer([0.5, 0.5], [[1.]]))
    ret = np.log(np.multiply.outer(initial_image_prior,
                                   np.multiply.outer(initial_rationality_prior,
                                                     initial_speaker_prior)))
    print("np.log() ", ret)
    print(np.log([[[0.5]], [[0.5]]]))

    ################################
