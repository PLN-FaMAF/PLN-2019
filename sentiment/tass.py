from xml.etree import ElementTree


class InterTASSReader:

    def __init__(self, filename, res_filename=None):
        self.filename = filename
        self.res_filename = res_filename
        self.root = ElementTree.parse(filename).getroot()

    def tweets(self):
        """Iterator over the tweets."""
        for tweet_el in self.root:
            assert len(tweet_el) == 6
            attrs = ['tweetid', 'user', 'content', 'date', 'lang']
            tweet = {}
            for attr in attrs:
                tweet[attr] = tweet_el.find(attr).text
            # now the sentiment
            tweet['sentiment'] = tweet_el.find('sentiment')[0][0].text

            yield tweet

    def X(self):
        """Iterator over the tweet contents."""
        for tweet_el in self.root:
            assert len(tweet_el) == 6
            content = tweet_el.find('content').text
            yield content

    def y(self):
        """Iterator over the tweet polarities."""
        if self.res_filename is None:
            # development dataset
            for tweet_el in self.root:
                assert len(tweet_el) == 6
                sentiment = tweet_el.find('sentiment')[0][0].text
                yield sentiment
        else:
            # test dataset
            with open(self.res_filename, 'r') as f:
                for line in f:
                    sentiment = line.split()[-1]
                    yield sentiment
