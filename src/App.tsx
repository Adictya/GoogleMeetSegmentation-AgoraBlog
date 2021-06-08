import React, { useEffect, useState } from 'react';
import './App.css';
import { generateDefaultGoogleMeetSegmentationTFLiteParams, generateGoogleMeetSegmentationTFLiteDefaultConfig, GoogleMeetSegmentationTFLiteWorkerManager } from "@dannadori/googlemeet-segmentation-tflite-worker-js"
import { GoogleMeetSegmentationTFLiteConfig, GoogleMeetSegmentationTFLiteOperationParams } from '@dannadori/googlemeet-segmentation-tflite-worker-js/dist/const';


interface WorkerProps {
    manager: GoogleMeetSegmentationTFLiteWorkerManager
    params : GoogleMeetSegmentationTFLiteOperationParams
    config : GoogleMeetSegmentationTFLiteConfig
}

const App = () => {
    const [workerProps, setWorkerProps] = useState<WorkerProps>()

    useEffect(()=>{
        const init = async () =>{
            const m = workerProps? workerProps.manager : new GoogleMeetSegmentationTFLiteWorkerManager()
            const c = generateGoogleMeetSegmentationTFLiteDefaultConfig()
            c.processOnLocal = true
            c.modelPath = "./meet.tflite"
            await m.init(c)

            const p = generateDefaultGoogleMeetSegmentationTFLiteParams()
            p.processWidth  = 256
            p.processHeight = 256
            p.kernelSize    = 0
            p.useSoftmax    = true
            p.usePadding    = false
            p.threshold     = 0.5
            p.useSIMD       = false
            const newProps = {manager:m, config:c, params:p}
            setWorkerProps(newProps)
        }
        init()
    }, [])

    useEffect(()=>{
        const input = document.getElementById("input")
        resizeDst(input!)
    })

    const resizeDst = (input:HTMLElement) =>{
        const cs = getComputedStyle(input)
        const width = parseInt(cs.getPropertyValue("width"))
        const height = parseInt(cs.getPropertyValue("height"))
        const dst = document.getElementById("output") as HTMLCanvasElement

        [dst].forEach((c)=>{
            c.width = width
            c.height = height
        })
    }

    useEffect(()=>{
        console.log("[Pipeline] Start", workerProps)
        let renderRequestId:number
        const render = async () => {
            console.log("pipeline...", workerProps)
            if(workerProps){
                console.log("pipeline...1")
                const src = document.getElementById("input") as HTMLImageElement
                const dst = document.getElementById("output") as HTMLCanvasElement
                const tmp = document.getElementById("tmp") as HTMLCanvasElement
                let prediction = await workerProps.manager.predict(src!, workerProps.params)

                const res = new ImageData(workerProps.params.processWidth, workerProps.params.processHeight)
                for(let i = 0;i < workerProps.params.processWidth * workerProps.params.processHeight; i++){
                    res.data[i * 4 + 0] = 0
                    res.data[i * 4 + 1] = 0
                    res.data[i * 4 + 2] = 0
                    res.data[i * 4 + 3] = prediction![i]
                }

                tmp.width  = workerProps.params.processWidth
                tmp.height = workerProps.params.processHeight
                tmp.getContext("2d")!.putImageData(res, 0, 0)

                dst.getContext("2d")!.clearRect(0, 0, dst.width, dst.height)
                dst.getContext("2d")!.drawImage(tmp, 0, 0, dst.width, dst.height)

                renderRequestId = requestAnimationFrame(render)
            }
        }
        render()
        return ()=>{
            cancelAnimationFrame(renderRequestId)
        }
    }, [workerProps])


    return (
        <div>
            <div style={{display:"flex"}}>
                <div style={{display:"flex"}}>
                    <img    width="300px" height="300px" id="input" src="srcImage.png"></img>
                    <canvas width="300px" height="300px" id="output"></canvas>
                    <canvas width="300px" height="300px" id="tmp" hidden></canvas>
                </div>
            </div>
        </div>
        );
}

export default App;
