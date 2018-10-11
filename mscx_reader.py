# -*- coding: cp949 -*-


from xml.dom import minidom
import codecs
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-path", "--filePath", type=str, default='./mscx_files/', help="file path")

args = parser.parse_args()




def print_nodes_recursive(xmldoc):
    for child in xmldoc.childNodes:
        print(child)
        print_nodes_recursive(child)



def delete_invisible_elements(xmldoc, tagname):
    element_list = xmldoc.getElementsByTagName(tagname)
    for elem in element_list:
        # print(vars(dyn))
        check_visible = elem.getElementsByTagName('visible')
        if not check_visible ==[]:
            # print(test)
            elem.parentNode.removeChild(elem)

    return xmldoc



def ommit_invisible_and_save(filepath):
    mscx_file = open(filepath, 'r')
    xmldoc = minidom.parse(mscx_file)
    mscx_file.close()
    outname = filepath[:-5] + '_cleaned.mscx'


    target_tags = ['Dynamic', 'Tempo', 'Pedal']


    for tag in target_tags:
        xmldoc = delete_invisible_elements(xmldoc, tag)

    with codecs.open(outname, "w", "utf-8") as out:
        xmldoc.writexml(out)


path = args.filePath

ommit_invisible_and_save(path)