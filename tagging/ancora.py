from nltk.corpus.reader.api import SyntaxCorpusReader
from nltk.corpus.reader import xmldocs
from nltk import tree
from nltk.util import LazyMap, LazyConcatenation


def parsed(element):
    """Converts a 'sentence' XML element (xml.etree.ElementTree.Element) to
    an NLTK tree.

    element -- the XML sentence element (or a subelement)
    """
    if element:
        # element viewed as a list is non-empty (it has subelements)
        subtrees = map(parsed, element)  # recursive call here!
        subtrees = [t for t in subtrees if t is not None]
        return tree.Tree(element.tag, subtrees)
    else:
        # element viewed as a list is empty. we are in a terminal.
        if element.get('elliptic') == 'yes' and not element.get('wd'):
            return None
        else:
            return tree.Tree(element.get('pos') or element.get('ne') or 'unk',
                             [element.get('wd')])


def tagged(element):
    """Converts a 'sentence' XML element (xml.etree.ElementTree.Element) to
    a tagged sentence.

    element -- the XML sentence element (or a subelement)
    """
    # http://www.w3schools.com/xpath/xpath_syntax.asp
    # XXX: XPath '//*[@wd]' not working
    # return [(x.get('wd'), x.get('pos') or x.get('ne'))
    #         for x in element.findall('*//*[@wd]')] + [('.', 'fp')]

    # convert to tree and get the tagged sent
    pos = parsed(element).pos()
    # filter None words (may return an emtpy list)
    return list(filter(lambda x: x[0] is not None, pos))


def untagged(element):
    """Converts a 'sentence' XML element (xml.etree.ElementTree.Element) to
    a sentence.

    element -- the XML sentence element (or a subelement)
    """
    # http://www.w3schools.com/xpath/xpath_syntax.asp
    # XXX: XPath '//*[@wd]' not working
    # return [x.get('wd') for x in element.findall('*//*[@wd]')] + [('.', 'fp')]

    # convert to tree and get the sent
    sent = parsed(element).leaves()
    # filter None words (may return an emtpy list)
    return list(filter(lambda x: x is not None, sent))


class AncoraCorpusReader(SyntaxCorpusReader):

    def __init__(self, path, files=None):
        if files is None:
            files = '.*\.tbf\.xml'
        self.xmlreader = xmldocs.XMLCorpusReader(path, files)

    def parsed_sents(self, fileids=None):
        return LazyMap(parsed, self.elements(fileids))

    def tagged_sents(self, fileids=None):
        return LazyMap(tagged, self.elements(fileids))

    def sents(self, fileids=None):
        return LazyMap(untagged, self.elements(fileids))

    def elements(self, fileids=None):
        # FIXME: skip sentence elements that will result in empty sentences!
        if not fileids:
            fileids = self.xmlreader.fileids()
        # xml() returns a top element that is also a list of sentence elements
        return LazyConcatenation(self.xmlreader.xml(f) for f in fileids)

    def tagged_words(self, fileids=None):
        return LazyConcatenation(self.tagged_sents(fileids))

    def __repr__(self):
        return '<AncoraCorpusReader>'


class SimpleAncoraCorpusReader(AncoraCorpusReader):
    """Ancora corpus with simplified Stanford CoreNLP POS tagset.

    https://nlp.stanford.edu/software/spanish-faq.shtml#tagset
    """

    def __init__(self, path, files=None):
        super().__init__(path, files)

    def tagged_sents(self, fileids=None):
        def f(s): return [(w, simple_tag(t)) for w, t in s]
        return LazyMap(f, super().tagged_sents(fileids))

    def parsed_sents(self, fileids=None):
        def f(t):
            for p in t.treepositions('leaves'):
                if len(p) > 1:
                    tag = t[p[:-1]].label()
                    t[p[:-1]].set_label(simple_tag(tag))
            return t

        return LazyMap(f, super().parsed_sents(fileids))


def simple_tag(t):
    """
    Convert general AnCora POS tag to tag in the simplified POS tagset used in
    Stanford CoreNLP.

    https://nlp.stanford.edu/software/spanish-faq.shtml#tagset
    """
    if t.startswith('a'):
        # assert t[1] in 'oq'
        return t[:2] + '0000'
    elif t.startswith('d'):
        # assert t[1] in 'adeinpt'
        return t[:2] + '0000'
    elif t.startswith('f'):
        # assert t in ['f0', 'faa', 'fat', 'fc', 'fd', 'fe', 'fg', 'fh', 'fia',
        #              'fit', 'fp', 'fpa', 'fpt', 'fs', 'ft', 'fx', 'fz']
        # (f0 and ft unobserved in ancora-3.0.1es)
        return t
    elif t in ['cc', 'cs', 'i', 'w', 'zm', 'zu']:
        return t
    elif t.startswith('nc'):
        return 'nc0{}000'.format(t[3])
    elif t.startswith('np'):
        return 'np00000'
    elif t.startswith('p'):
        # assert t[1] in '0deinprtx'
        return t[:2] + '000000'
    elif t.startswith('r'):
        # assert t in ['rg', 'rn']
        return t
    elif t.startswith('sp'):
        return 'sp000'
    elif t.startswith('v'):
        # 34 possibilities for t[:4]
        return t[:4] + '000'
    elif t.startswith('z'):
        # assert t in ['z', 'zp']
        # ('zm' and 'zu' were already tested)
        return 'z0'
    else:
        # not a valid POS: named entity ('ne' field) or 'unk'
        return t
