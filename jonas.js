console.log('jonas.js loaded')

jonas=function(){
    console.log('jonas fun initialized at '+Date())
}

jonas.getIris=async function(){
    jonas.irisData=await (await fetch('https://episphere.github.io/ai/data/iris.json')).json()
    return jonas.irisData
}