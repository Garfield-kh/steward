// Emit deterministic tool definitions, model-visible outputs and calibration for profile comparisons.
import { registerHooks, stripTypeScriptTypes } from 'node:module'
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
const stub = source => `data:text/javascript,${encodeURIComponent(source)}`
registerHooks({
  resolve(specifier, context, next) {
    if (specifier === '@deepseek-ai/dsh-tools') return { url: stub('export const defineTool=x=>x'), shortCircuit:true }
    if (specifier === '@steward/dsh-ego') return { url:stub('export const recordingId=x=>x; export class EgoError extends Error {}'), shortCircuit:true }
    return next(specifier,context)
  },
  load(url,context,next) {
    if (url.endsWith('.ts')) return { format:'module',shortCircuit:true,source:stripTypeScriptTypes(readFileSync(new URL(url),'utf8'),{mode:'transform'}) }
    return next(url,context)
  }
})
const {egoTools}=await import(pathToFileURL(process.argv[2]))
const {parseCameraCalibration}=await import(pathToFileURL(process.argv[3]))
const receipt={id:'receipt',skill:'fasten_onto',feasible:true,risk:0.1,maxRisk:0.35,withinRiskLimit:true,
  issues:[],expiresAt:12,plan:{target:'part',support:'table',pathLength:2,reach:0.3,drive:'none',
  seatTarget:{targetSource:'observed'},releaseGapM:0.001,navigationEvidence:{reason:'unobserved'}},
  nextAction:{skill:'fasten_onto',params:{part:'part',body:'body'}},warnings:[{reason:'unverified',nextAction:{skill:'observe'}}]}
const job={id:'job',skill:'fasten_onto',receipt:'receipt',state:'succeeded',message:'executed',log:['one','two'],
  postcondition:{executed:true,verified:false,released:1,confirmed:0,required:2,actionStatus:'continue',
  nextAction:{skill:'fasten_onto'},holeProgress:[{hole:'h1',screw:'s1',released:true,status:'executed_unverified'}]}}
const ego={summary:async()=>({objects:[],held:null}),predict:async()=>receipt,job:async()=>job,run:async()=>job,wait:async()=>job}
const tools=egoTools(ego), outputs={}
for (const name of ['ego_state','ego_predict','skill_status','skill_run']) {
  const tool=tools.find(t=>t.name===name),args={skill:'fasten_onto',params:{part:'part',body:'body'},job:'job',receipt:'receipt',wait:false}
  const value=await tool.execute(args,{signal:new AbortController().signal})
  outputs[name]=tool.output.render(args,value)
}
const transform=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
const view={width:640,height:480,K:[[500,0,320],[0,500,240],[0,0,1]],camera_to_reference:transform,
 renderNearM:0.02,depthRangeM:[0.02,0.5],housingSizeM:[0.016,0.016,0.012],tipPlaneClearanceM:0.009}
const calibration=parseCameraCalibration({reference_camera:'head_camera',calibration_source:'rig_calibration',
 camera_axes:'opencv_x_right_y_down_z_forward',stamp:12,joint_encoders:{lift:0.4},views:{head_camera:view,wide_camera:view}})
console.log(JSON.stringify({tools:tools.map(t=>({name:t.name,description:t.description,parameters:t.parameters,schema:t.output.schema})),outputs,calibration}))
