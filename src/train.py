from datasets import FbDataset
from topic_models import GensimLDA
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='FbBNPostTopicModeling')
# parser.add_argument('--model', type=str, default='LDA',
#                     help='[LDA]')
parser.add_argument('--train', action="store_true",
                    help='wheter to train the model or not')
args = parser.parse_args()


def infer(posts):
    """

    :param documents: list of tuples of posts(post_id, post_text)
    """

    # posts = [(0,'করোনার বিস্তারিত চিকিৎসা করোনা ভাইরাস কি? করোনা ভাইরাস , সেই সকল ভাইরাস পরিবারের সদস্য যারা আমাদের স্বাভাবিক ঠান্ডা কাশি থেকে শুরু করে SARS( Severe Acute Respiratory Syndrome) ও MERS(Middle East Respiratory Syndrome) করে থাকে। এই ভাইরাস পরিবারের সর্ব শেষ আবিষ্কৃত সদস্য হচ্ছে কোভিড ১৯ যেটা মানব দেহে আগে কখনো আগে দেখা যায় নাই।...Continue Reading')
    #     , (1, 'ঠিক যেন পরমাণু বিস্ফোরণ ঘটল, লেবাননে বহু হতাহত (ভিডিও)'), (2, 'মেজর সিনহাকে নিয়ে তার মায়ের বক্তব্য ভাইরাল')]

    dataset = FbDataset(posts)
    preprocessed = dataset.preprocess()
    model = GensimLDA()
    model.load_model('../models/topic-model')
    prediction = model.predict_topics(preprocessed)
    print(prediction)
    ids = [id for id, text in posts]

    res = list(zip(ids, prediction))
    topic_df = model.format_topics_sentences(preprocessed, posts)
    # print('prediction', res)
    print( topic_df)

def train():
    data = pd.read_csv('../data/fb_data_scraped.csv')[['Column0', 'Column5']]
    data.dropna(inplace=True)
    print(data.head())
    posts = data.to_numpy().squeeze()
    # print('posts', posts)
    dataset = FbDataset(posts)
    preprocessed = dataset.preprocess()
    model = GensimLDA()
    model.fit(dataset, 20)
    model.save('../models/topic-model')
    # model = GensimLDA()
    # model.load_model('./topic-model')
    prediction = model.predict_topics(preprocessed[0:5])
    print(prediction)


if __name__ == "__main__":
    # Loading the models.
    # model_name = args.model

    if args.train:
        train()

    posts = [(0,'করোনার বিস্তারিত চিকিৎসা করোনা ভাইরাস কি? করোনা ভাইরাস , সেই সকল ভাইরাস পরিবারের সদস্য যারা আমাদের স্বাভাবিক ঠান্ডা কাশি থেকে শুরু করে SARS( Severe Acute Respiratory Syndrome) ও MERS(Middle East Respiratory Syndrome) করে থাকে। এই ভাইরাস পরিবারের সর্ব শেষ আবিষ্কৃত সদস্য হচ্ছে কোভিড ১৯ যেটা মানব দেহে আগে কখনো আগে দেখা যায় নাই।...Continue Reading')
        , (1, 'ঠিক যেন পরমাণু বিস্ফোরণ ঘটল, লেবmodelsাননে বহু হতাহত (ভিডিও)'), (2, 'মেজর সিনহাকে নিয়ে তার মায়ের বক্তব্য ভাইরাল')]

    infer(posts)



