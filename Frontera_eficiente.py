import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

with st.sidebar.form(key='my_form'):
    st.write("""
        ## Frontera de Cartera Eficiente
        Agregar tickers separados por una coma.
        Ejemplo: ko,mcd,tsla
        """)
    tickerSymbol = st.text_input('Ingresa los ticker:')
    desde = st.date_input('Datos desde')
    hasta = st.date_input('Datos hasta')
    q = st.number_input('Cantidad de pruebas para optimización',min_value=100, max_value=10000) 
    mc = st.checkbox("Matriz de correlaciones",value=False)
    bp = st.checkbox("Calculo de PER y BETA",value=False)
    submit_button = st.form_submit_button(label='Submit')

if tickerSymbol:
    lista_ticker = tickerSymbol.split(',')
    if len(lista_ticker)>1:
        try:
            st.write("""# Frontera de Cartera Eficiente""")

            

            df=yf.download(lista_ticker,start=desde,end=hasta)['Adj Close']
            df=df.loc[~(df==0).any(axis=1)]
            ret_log=np.log((df/df.shift(1)).dropna())
            #ret_log

            def sharpe_neg(weights):
                global ret_log
                weights = np.array(weights)
                ret = np.sum(ret_log.mean()*weights)*len(ret_log)
                vol = np.sqrt(np.dot(weights.T,np.dot(ret_log.cov()*len(ret_log),weights)))
                sr = ret/vol
                return -sr

            def check_sum(weights):
                return np.sum(weights)-1

            cons = ({'type':'eq','fun':check_sum})
            bounds=[(0,1)]*len(ret_log.columns)
            init_guess = [0.1]*len(ret_log.columns)
            from scipy import optimize
            opt_result= optimize.minimize(sharpe_neg,init_guess,bounds=bounds,constraints=cons)
            
            st.write("""
            ## Resultados de la optimización
            """)
            opt_result

            def optimizar(data,q=200):
                pond=np.array(np.random.random(len(data.columns)))
                pond=pond/np.sum(pond)
                carteras=[]
                for i in range(q):
                    pond=np.array(np.random.uniform(0,1,len(data.columns)))
                    pond=pond/np.sum(pond)
                    r={}
                    r['retorno']=np.sum((data.mean()*pond*len(data)))
                    r['volatilidad']=np.sqrt(np.dot(pond,np.dot(data.cov()*len(data),pond)))
                    r['sharpe']=r['retorno']/r['volatilidad']
                    r['pesos']=pond.round(4)
                    carteras.append(r)
                carteras = pd.DataFrame(carteras)
                return carteras.sort_values('sharpe', ascending=False).reset_index()

            
            carteras=optimizar(ret_log,q=q)


            st.write(f"## Carteras testeadas : {q}")
            carteras

            datos_tickers=[]
            for ticker in ret_log.columns:
                d={}
                d['ticker']=ticker
                d['retorno']=ret_log[ticker].mean()*len(ret_log)
                d['volatilidad']=ret_log[ticker].std()*(len(ret_log)**0.5)
                d['sharpe']=d['retorno']/d['volatilidad']
                if bp:
                    tickerData = yf.Ticker(ticker).info
                    d['beta'] = tickerData['beta']
                    d['per'] = tickerData['trailingPE']
                datos_tickers.append(d)

            datos_tickers = pd.DataFrame(datos_tickers).set_index('ticker')
            mejor_port = carteras.iloc[carteras['sharpe'].idxmax()]['pesos']

            datos_tickers['ponderacion optima']=mejor_port
            datos_tickers = datos_tickers.fillna(0)
            datos_tickers = datos_tickers.sort_values('ponderacion optima',ascending=False)
            if not bp:
                datos_tickers.loc['Total']=[
                    carteras.iloc[carteras['sharpe'].idxmax()]['retorno'],
                    carteras.iloc[carteras['sharpe'].idxmax()]['volatilidad'],
                    carteras.iloc[carteras['sharpe'].idxmax()]['sharpe'],
                    1
                ]
            else:
                datos_tickers.loc['Total']=[
                carteras.iloc[carteras['sharpe'].idxmax()]['retorno'],
                carteras.iloc[carteras['sharpe'].idxmax()]['volatilidad'],
                carteras.iloc[carteras['sharpe'].idxmax()]['sharpe'],
                sum(datos_tickers['beta']*datos_tickers['ponderacion optima']),
                sum(datos_tickers['per']*datos_tickers['ponderacion optima']),
                1
                ]

            st.write("## Datos de los activos estudiados")
            datos_tickers


            optimo={}
            #print(carteras.iloc[carteras['sharpe'].idxmax()])
            optimo['retorno']=carteras.iloc[carteras['sharpe'].idxmax()]['retorno']
            optimo['volatilidad']=carteras.iloc[carteras['sharpe'].idxmax()]['volatilidad']
            optimo['sharpe']=carteras.iloc[carteras['sharpe'].idxmax()]['sharpe']

            st.write("## Resultados de la Cartera Eficiente")
            optimo

            #FRONTERA OPTIMA
            lista_volatilidad = np.linspace(carteras.volatilidad.min(),carteras.volatilidad.max(),30)

            frontera=[]
            for v in lista_volatilidad:
                vr={}
                vr['volatilidad']=v
                vr['retorno']=carteras.loc[(carteras.volatilidad<v),'retorno'].max()
                frontera.append(vr)
            frontera= pd.DataFrame(frontera)

            #grafico
            def grafico():
                fig = plt.figure(figsize=(7,6),dpi=500)
                plt.scatter(carteras.volatilidad,carteras.retorno,c=carteras.sharpe,s=1,cmap='rainbow')
                plt.colorbar(label='Sharpe Ratio',aspect=40)
                plt.xlabel('Volatilidad')
                plt.ylabel('Retorno')
                str_tickers = ",".join(list(ret_log.columns))
                titulo = 'Frontera de cartera eficiente \n('+ str_tickers+')'
                plt.title(titulo)
                plt.scatter(optimo['volatilidad'],optimo['retorno'],c='tab:red',alpha=0.2,s=1500)
                plt.text(optimo['volatilidad'],optimo['retorno'],'Optimo',fontsize=9,c='k',ha='center',va='center')

                texto_optimo = ""
                for ticker in ret_log.columns:
                    vol=datos_tickers.loc[ticker,'volatilidad']
                    ret=datos_tickers.loc[ticker,'retorno']
                    plt.scatter(vol,ret,c='tab:blue',s=500)
                    plt.text(vol,ret,ticker,c='w',ha='center',va='center',fontsize=4)
                    texto_optimo+=(ticker+": "+str(round(datos_tickers.loc[ticker,'ponderacion optima']*100,2))+"%\n")
                texto_optimo = 'Cartera Optima:\n'+texto_optimo
                #plt.text(optimo['volatilidad']*1.1,optimo['retorno'],texto_optimo,fontsize=9,c='k',ha='center',va='center')
                #plt.xlim(carteras['volatilidad'].min(),carteras['volatilidad'].max())
                return fig , texto_optimo
            fig_fce, cartera = grafico()
            
            
            st.write(fig_fce)
            st.write(cartera)
            
            if mc:
                plt.clf()
                def matriz_corr():
                    C = df.corr()
                    fig_mc = sns.heatmap(C, annot = True)
                    plt.title(f"HeatMap Correlación entre {lista_ticker}")
                    return fig_mc
                matriz=matriz_corr()
                st.write("## Matriz de Correlación")
                st.pyplot(matriz.figure)


            

            
            st.write("""
            
            Elaborado por Francisco Carreño

            """)
        except:
            st.write("""
                # Ocurrió un error durante el proceso, revisá la lista de tickers por favor
                """)
    else:
        st.write("""
        # Debes ingresar más de 1 ticker válido
        """)