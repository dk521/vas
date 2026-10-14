import numpy as np

class SimilarityService:

    @staticmethod
    def consine_similarity(embedding1, embedding2):

        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

        return similarity