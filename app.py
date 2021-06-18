import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import cv2
# import matplotlib.pyplot as plt
# import plotly.express as px
import tensorflow as tf
import os
import datetime
#os.chdir('C:/Users/nithi/OneDrive/Desktop/Our project')



app = dash.Dash(__name__,update_title = None,title ='Brain-Tumor-Segmentation', external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

server = app.server

a = 'No image uploaded!'


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    html.Div(id='live-update-text',style={'font-size':'20px'}),
    dcc.Interval(id='interval-component',interval=1*1000,n_intervals=0),
   
    html.Div(id='page-content'),
   
    dbc.Row([
        
        dbc.Col([html.Button('Home',id = 'btn_1', n_clicks=0,
                className='button button4')]),
        dbc.Col([html.Button('Results',id = 'btn', n_clicks=0,className='button button4')]),
    
        dbc.Col([html.Button('Generate Report',id='js',n_clicks=0,  className='button button4')])

       ]),
    
    html.Br(),
    html.Div(id='simiple'),
    
    dcc.Store(id = 'name'),
    dcc.Store(id = 'age'),
    dcc.Store(id = 'gender'),
    dcc.Store(id = 'referee'),
    dcc.Store(id = 'image'),
    dcc.Store(id = 'pred_image'),
],className='column',
style={
  'verticalAlign':'middle',
  #'textAlign': 'center',
  'background-image':'url("assets/image.jpg")',
  'position':'fixed',
  'background-size': 'cover',
  'width':'100%',
  'height':'100%',
  'top':'0px',
  'left':'0px',
  'overflow': 'scroll',
  'padding-bottom': '10%'
  #'z-index':'1000'
})

card_content1=[
    dbc.CardHeader("Patient Info"),
    dbc.CardBody([

        dbc.Input(id = 'input_name', placeholder = 'Enter Name', type = 'text'),
        html.Br(),
        dbc.Input(id = 'input_age', placeholder = 'Enter Age', type = 'number'),
        html.Br(),
        dbc.FormGroup(
            [
             dbc.Label("Select the Gender"),
             dbc.RadioItems(options=[
            {"label":"Male","value":"Male"},
            {"label":"Female","value":"Female"},
            {"label":"Transgender","value":"Transgender"}],value=[],id="input_gender",inline=True,),
            ]),
        html.Br(),
        dbc.Input(id = 'input_referee', placeholder = 'Referred By', type = 'text'),
       
        ])]

card_content2 = [
    dbc.CardHeader("Upload Image"),
    dbc.CardBody(
        [
            dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                         html.A('Select Image',style={'text-decoration':'underline','color':'blue','cursor':'pointer'})
                    ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
        },
        multiple=False
        ),
        
        dbc.Spinner([
                html.Img(id='output-image-upload', style={'height':'40%', 'width':'40%','margin': '0 auto','display': 'block',}),
                #html.Img(id='predicted_img', style={'height':'20%', 'width':'20%','display':'none'}),
                #html.Div(id='sample',style={'display':'none'}),
            ], color = 'dark', size = 'sm')]
    )]



index_page = html.Div([
        html.H1("Brain SCAT", style = {'textAlign':'center', 'color':'black', 'font-family':'Custom'}), 
        html.H3("Segmentation and Classification of Brain Tumors", style = {'textAlign':'center', 'color':'black', 'font-family':'Custom'}),
        html.Br(),
        html.Br(),
        dbc.CardDeck([

                (dbc.Card(card_content1, color="light")),
                (dbc.Card(card_content2, color="light")),             
])],style={"padding": "5%"})



patient_details_card = [
    dbc.CardHeader("Patient Details"),
    dbc.CardBody(
        [
            dbc.Row([
                dbc.Col(html.H5("Name: ", className="card-title")),
                dbc.Col(html.Div(id = 'display_name')),
                
                ]),
            dbc.Row([
                dbc.Col(html.H5("Referred by: ", className="card-title")),
                dbc.Col(html.Div(id = 'display_referee'))
                ]),
            dbc.Row([
                dbc.Col(html.H5("Age: ", className="card-title")),
                dbc.Col(html.Div(id = 'display_age')),
                ]),
            dbc.Row([
                dbc.Col(html.H5("Gender: ", className="card-title")),
                dbc.Col(html.Div(id = 'display_gender'))
                ]),
            
            html.Hr(),
            dbc.Row([
                dbc.Col(html.H5("Your Brain CT ", className="card-title"))
                ]),
            html.Br(),
            dbc.Row([
                dbc.CardImg(id = 'display_img', style = {'height': '250px', 'width': '250px' ,'margin': '0 auto','display': 'block'})
                ]),
            
        ]
    ),
]

jumbotron = dbc.Jumbotron(
    [
        dbc.Container(
            [
                html.P(
                    "Predicted Tumor Class", className="lead",
                ),
                html.H1(id = 'a', className="display-3"),
                html.Hr(className="my-2"),
                html.H3([dbc.Badge("Segmented CT Image", color = "info", className="ml-1")]),
                dbc.CardImg(id = 'display_predicted_img',  style = {'height': '250px' , 'width': '250px','margin': '0 auto','display': 'block'}),
            ],
            fluid=True,
        )
    ],
    fluid=True,
)


page_1_layout = html.Div([
    
     html.H1("Results", style = {'textAlign':'center', 'color':'black', 'font-family':'Custom'}),
     
     html.Div([
         html.H6(id = 'Date')
         ], style = {'float':'right'}),
     
     html.Br(),
     html.Br(),
     dbc.Row([
          
                dbc.Col(dbc.Card(patient_details_card, color="light"), width = 4, style={"height": "100%"}),
                dbc.Col(jumbotron, style={"height": "100%"}),
                
        ], justify = 'center'),
    
], id='printid',style = {'padding-left':'5%', 'padding-right':'5%', 'padding-top':'2%'})


def decode_img(string):
    decoded = base64.b64decode(string)
    buffer = BytesIO(decoded)
    im = Image.open(buffer)
    im = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (256, 256),interpolation = cv2.INTER_NEAREST)
    im = np.reshape(im,(1,256,256))
    im = im / 255.0
    return im

def make_prediction(img):
    interpreter = tf.lite.Interpreter(model_path='model1.tflite')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = [interpreter.get_output_details()[0]["index"],interpreter.get_output_details()[1]["index"]]
    sample_img = img.reshape([1,256,256,1]).astype('float32')
    
    interpreter.set_tensor(input_index,sample_img)
    interpreter.invoke()
    
    labels= interpreter.get_tensor(output_index[0])
    predictions= interpreter.get_tensor(output_index[1])
    
    class_names =  ['Meningioma','Glioma', 'Pituitary tumor']
    lab = int(np.argmax(labels))
    lab = class_names[lab]
    pred = np.where(predictions>0.5,255,0)
    return pred,lab


@app.callback(Output('Date','children'),
              [Input('interval-component','n_intervals')])
def update_date(n):
    return [html.P('Updated: '+str(datetime.datetime.now().strftime("%d-%m-%Y %H:%M")))]

@app.callback([Output('name', 'data'),
               Output('age', 'data'),
               Output('gender', 'data'),
               Output('referee', 'data'),],
              [Input('input_name', 'value'),
               Input('input_age', 'value'),
               Input('input_gender', 'value'),
               Input('input_referee', 'value'),
               ])
def storing(name, age, gender, referee):
    return name, age, gender, referee


@app.callback([Output('page-content', 'children'),
               Output('btn', 'hidden'),
               Output('btn_1', 'hidden'),
               Output('js', 'hidden'),
               Output('display_img', 'src'),
               Output('display_name', 'children'),
               Output('display_age', 'children'),
               Output('display_gender', 'children'),
               Output('display_referee', 'children'),
               Output('a', 'children'),
               Output('display_predicted_img','src')],
              [Input('btn', 'n_clicks'),
               Input('btn_1', 'n_clicks')],
              [State('image', 'data'),
              State('name', 'data'),
              State('age', 'data'),
              State('gender', 'data'),
              State('referee', 'data'),
              State('pred_image','data')])
def display_page(n_clicks, n_clicks1, img, name, age,gender, referee, pred_img):
    trigger = dash.callback_context.triggered[0]
    if (trigger['prop_id'].split('.')[0] == 'btn'):
        return page_1_layout, True, False,False, img, name, age,gender, referee, a, pred_img
    elif (trigger['prop_id'].split('.')[0] == 'btn_1'):
        return index_page, False, True,True,  dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    else:
        return index_page, False, True,True, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update



@app.callback([Output('image', 'data'),
               Output('output-image-upload', 'src'),
               Output('pred_image', 'data'),
               Output('sample', 'children')],
              [Input('upload-image', 'contents')])
def update_output(list_of_contents):
    if list_of_contents is not None:
        image = decode_img(list_of_contents.split(',')[1])
        
        masks,label = make_prediction(image)
        global a
        a = label
        x = np.reshape(image,(256,256))*255.0
        y = np.reshape(masks,(256,256))
        
        xpro = Image.fromarray(np.uint8(x))
        ypro = Image.fromarray(np.uint8(y))
        newimg = Image.blend(xpro, ypro, alpha=0.6)
        blendedimg = np.asarray(newimg)
        
        _, buffer1 = cv2.imencode('.png', x)
        encstring1 =  'data:image/png;base64,{}'.format(base64.b64encode(buffer1).decode('utf-8'))
        
        _, buffer = cv2.imencode('.png', blendedimg)
        encstring =  'data:image/png;base64,{}'.format(base64.b64encode(buffer).decode('utf-8'))
        
        return  encstring1,encstring1,encstring,label
    
    else:
        return dash.no_update,dash.no_update,dash.no_update,dash.no_update

app.clientside_callback(
    """
    function(n_clicks){
        if(n_clicks > 0){
            var opt = {
                margin: 1,
                filename: 'report.pdf',
                image: { type: 'jpeg', quality: 0.98 },
                html2canvas: { scale: 3},
                jsPDF: { unit: 'cm', format: 'a2', orientation: 'p' },
                pagebreak: { mode: ['avoid-all'] }
            };
            html2pdf().from(document.getElementById("printid")).set(opt).save();
        }
    }
    """,
    Output('js','value'),
    Input('js','n_clicks')
)

        


if __name__ == '__main__':
    app.run_server(debug=False)
