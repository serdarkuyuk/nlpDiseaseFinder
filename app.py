import os
import streamlit as st
import pickle
import spacy
from spacy import displacy
import en_core_web_sm
import os


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


def load_model(model_path):
    ''' Loads a pre-trained model for prediction on new test sentences

    model_path : directory of model saved by spacy.to_disk
    '''
    nlp = spacy.blank('en')
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    ner = nlp.from_disk(model_path)

    return ner


def findout_disease(text):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'nermodel')

    ner = load_model(str(filename)+"/")
    mydoc = ner(text)

    return mydoc


def welcome():
    return "Welcome All"


def example():

    example_text = "The authors describe the case of a 56 - year - old woman with chronic , severe heart failure secondary to dilated cardiomyopathy and absence of significant ventricular arrhythmias who developed QT prolongation and torsade de pointes ventricular tachycardia during one cycle of intermittent low dose ( 2 . 5 mcg / kg per min "

    return example_text


def main():
    st.title("Disease Finder")

    st.text('This app finds disease names from a text.  For detail information follow the below link.')
    link = '[GitHub](https://github.com/serdarkuyuk/nlpDiseaseFinder)'
    st.markdown(link, unsafe_allow_html=True)

    st.header('Example how to use this website')
    example_text = example()

    st.text("Let's assume you have this text ")
    example_text = st.text_area("Example text", example_text)

    if st.button("Find Disease"):

        st.text('The model has an answer')
        example_text = example()
        mydoc = findout_disease(example_text)
        html = displacy.render(mydoc, style="ent")

        html = html.replace("\n", " ")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    if st.button("Locations"):
        example_text = example()
        mydoc = findout_disease(example_text)
        for ent in mydoc.ents:
            st.text([ent.text, ent.start_char, ent.end_char, ent.label_])

        st.text('If you like it please send an email to serdarkuyuk@gmail.com')

    st.header('Now it is your turn')
    yourtext = st.text_area("Example text", "Type your sentence here.")

    if st.button("Find Your Disease"):

        mydoc2 = findout_disease(yourtext)
        st.text('The model has an answer')

        html = displacy.render(mydoc2, style="ent")

        html = html.replace("\n", " ")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    if st.button("Location"):
        mydoc2 = findout_disease(yourtext)
        for ent in mydoc2.ents:
            st.text([ent.text, ent.start_char, ent.end_char, ent.label_])


if __name__ == '__main__':
    # nlp = en_core_web_sm.load()
    nlp = spacy.blank('en')

    # ner = nlp.from_disk('/Users/serdar/Documents/udel/python/nlpDiseaseFinder/nermodel/')
    # ner = nlp.from_disk(str(filename)+'/')

    #customNlp = pickle.load(open("nlp.pikcle", "rb"))
    # spacy.load(customNlp)
    main()
