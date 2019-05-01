import os
import csv
import xml.etree.ElementTree as ElementTree
from algorithm.helpers.generator import Generator


class Task:
    def __init__(self, path, reader=None):
        self.path = path
        if reader:
            self.reader = reader
        else:
            self.reader = Task.text_only_xml

    def get_split(self, name, part, chunks, start=1):
        path = os.path.join(self.path, 'splits', name, '%s.csv' % part)
        return self.generator(path, chunks, start=1)

    def generator(self, truth_file, chunks, start=1):
        """
        Returns a generator for a given truth file and a chunk
        :param truth_file: File containing the IDs and labels
        :param chunks: Number of the desired chunk
        :return: generator : subjectID, label, content for chunk
        """
        with open(truth_file) as f:
            truth = []
            for row in csv.reader(f, delimiter='\t'):
                if len(row) == 2:
                    paths = [os.path.join(self.path, 'chunks', 'chunk{}'.format(chunk),
                                          '{}_{}.xml'.format(row[0], chunk))
                             for chunk in range(start, chunks + 1)]
                    truth.append((row[0], row[1], paths))
            f.close()
        return Generator(truth, self.reader)

    @staticmethod
    def text_only_xml(path):
        """
        Retrieves the text (titles and content) from a user's writings at a given chunk. Pass this function
        as a parameter to __init__
        :param path: path to xml file
        :return: concatenated titles (from empty posts) and contents
        """
        with open(path) as file:
            tree = ElementTree.parse(file)
            writings = []
            for writing in tree.findall('.//WRITING'):
                content = writing.find('TITLE').text.strip()
                content += writing.find('TEXT').text.strip()
                writings.append(content)
            return reversed(writings)

    @staticmethod
    def with_dates_xml(path):
        """
        Retrieves the text (titles and content) and dates from a user's writings at a given chunk. Pass this function
        as a parameter to __init__
        :param path: path to xml file
        :return: concatenated dates, titles (from empty posts) and contents
        """
        with open(path) as file:
            tree = ElementTree.parse(file)
            writings = []
            for writing in tree.findall('.//WRITING'):
                date = writing.find('DATE').text.strip()
                text = writing.find('TITLE').text.strip()
                text += writing.find('TEXT').text.strip()
                writings.append({'date': date, 'text': text})
            return reversed(writings)
