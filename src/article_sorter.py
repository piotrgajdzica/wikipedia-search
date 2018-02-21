import os


def save_article(text, filename):
    f = open(filename, 'w', encoding='UTF-8')
    f.write(text)
    f.close()


def get_articles(data_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    article_id = 0

    for file in os.listdir(data_dir):

        article = ""

        for line in open(os.path.join(data_dir, file), encoding='UTF-8').readlines():
            if not line.startswith("<doc") and not line.startswith("</doc") and line != "\n":
                article += line

            if line.startswith("</doc"):
                save_article(article, os.path.join(output_dir, str(article_id) + ".txt"))
                article = ""
                article_id += 1


def main():

    data_dir = "./data/simple_english/"
    output_dir = "./data/articles/"

    get_articles(data_dir, output_dir)


main()
