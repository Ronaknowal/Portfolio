"""Freeze the topic only after actual author checks and opened-image review."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
evidence=ROOT/'docs/teaching/evidence'
out=ROOT/'scratch/ito-sde-verification'
browser_dir=ROOT/'scratch/ito-sde-browser'
files=[
 'src/learn/data/topics/ito-calculus-stochastic-differential-equations.jsx',
 'src/learn/data/ito-sde-models.js',
 'src/learn/data/ito-sde-examples.js',
 'src/learn/components/lesson-labs/ItoSdeLabs.jsx',
 'src/learn/components/lesson-labs/ito-sde-labs.css',
 'src/learn/data/curriculum/blueprints/it-calculus-stochastic-differential-equations.js',
]
def fingerprint(path):
    data=(ROOT/path).read_bytes()
    return {'path':path,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
production=list(map(fingerprint,files))
native=json.loads((out/'native-results.json').read_text(encoding='utf-8'))
model=json.loads((out/'model-results.json').read_text(encoding='utf-8'))
browser=json.loads((browser_dir/'results.json').read_text(encoding='utf-8'))
reading=json.loads((browser_dir/'reading-results.json').read_text(encoding='utf-8'))
assert native['passed'] and model['passed']
assert model['sourceSha256']==production[1]['sha256']
assert native['sourceSha256']==production[2]['sha256']
assert [r['width'] for r in browser['results']]==[1440,390,320]
assert [r['states'] for r in browser['results']]==[318,319,319]
for row in browser['results']:
    assert row['geometry']['fonts'] and row['geometry']['documentWidth']==row['width']
    assert len(row['programs'])==13 and row['independentPractice']==11
    assert not row['errors'] and not row['warnings'] and not row['failedRequests']
    assert not row['geometry']['svgOverflow']
    assert all(e['content']<=e['width']+1 for e in row['geometry']['equations'])
for row in reading['results']:
    assert row['geometry']['fonts'] and not row['errors'] and not row['failed']
    assert len(row['geometry']['equations'])==21
    assert not row['geometry']['svgOverflow']
    assert all(e['content']<=e['width']+1 for e in row['geometry']['equations'])
    assert all(c['width']<=c['container']+1 and c['labelSize']>=14 for c in row['geometry']['compact'])

# Only include image names actually opened after the final browser captures.
opened=json.loads((out/'opened-final-images.json').read_text(encoding='utf-8'))
images=[fingerprint('scratch/ito-sde-browser/'+name) for name in opened]
now=datetime.now(timezone.utc).isoformat()
record={
 'topicId':'it-calculus-stochastic-differential-equations','modulePosition':37,
 'authorFrozenAt':now,'status':'author-verified; independent review and shared production integration pending',
 'production':production,'native':native,'model':model,'browser':browser,'finalReading':reading,
 'openedImages':images,'openedImagesNote':'Actual image-tool inspection by the author; images are finite visual evidence, not mathematical proofs.',
 'design':'docs/teaching/ITO-CALCULUS-SDE-LESSON-DESIGN.md',
 'verification':'docs/teaching/ITO-CALCULUS-SDE-VERIFICATION.md',
 'original':'docs/teaching/evidence/ito-calculus-original-content.json',
 'independentFindingsBeforeFreeze':[
   {'finding':'The QV paragraph and changed-increment answer called the refinement random.',
    'resolution':'Both now explicitly say random Brownian increments observed along deterministic partition refinements; the existing mesh/dyadic proof is retained.'},
   {'finding':'The GBM quantile uses sigma rather than its absolute value.',
    'resolution':'The body now declares nonnegative volatility before the formulas and explains the absolute-value scale for a negative written scalar diffusion coefficient.'},
 ],
 'readingRefinements':['Long equations broken into readable aligned steps without shrinking type.',
   'Simple analytic distributions/error curves, one-step stress plot and curvature figure fit mobile; dense paths retain announced keyboard scrolling.',
   'Long native-select captions shortened without removing the nearby Gaussian/atom explanations.',
   'Compact labels use20.5 source pixels and a larger left axis margin; the final interaction and fresh reading checks verify all three widths and every compact label at the frozen size.'],
 'limits':['Finite bounded teaching models, not arbitrary-range arithmetic.',
   'Selector values and representative boundary/interior time cursors checked; no claim to exhaust every seed or continuous input.',
   'Seeded paths, analytic laws and schematic geometry are explicitly distinguished.',
   'No screen-reader speech pass or untested-browser certification.',
   'The official video page/identity and companion notes were checked; no full-video watch claim.',
   'No author-owned shared registry, catalogue, global stylesheet, build or ledger mutation.'],
}
target=evidence/'ito-sde-author-review.json'
target.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
(out/'final-source-hashes.json').write_text(json.dumps({'authorFrozenAt':now,'production':production},indent=2),encoding='utf-8')

note=ROOT/'docs/teaching/topic-notes/it-calculus-stochastic-differential-equations.md'
text=note.read_text(encoding='utf-8').replace('- Status: open.','- Status: resolved — implemented and author-verified; independent review is tracked separately.',1)
text=text.replace("The note remains open until the destination's actual implementation and verification; see the [Itô/SDE design](../ITO-CALCULUS-SDE-LESSON-DESIGN.md).", "The destination is now implemented and author-verified; see the [Itô/SDE verification](../ITO-CALCULUS-SDE-VERIFICATION.md).")
text=text.replace('Destination implementation is still for its owner to assess; this note does not mark that topic complete.', 'Destination author checks are complete; independent review and shared integration remain separate.')
text += '\n\n## Destination implementation closure — 11 September 2026\n\nSections 1–3 now derive the information contract, finite QV identities, deterministic-partition L² bound, dyadic almost-sure scope and square-integrable integral construction. Section 7 groups the same fine Brownian increments for EM/Milstein/exact GBM; section 5 separately derives the correct weighted OU coupling. Section 10 retains the exact constant-coefficient crossing reminder and explains why a nonlinear-SDE bridge approximation needs separate validation. The proposal to use a midpoint is implemented as the explicitly defined symmetric endpoint average, avoiding ambiguity about sampling the temporal midpoint. The actual original/native, model and desktop/mobile/keyboard evidence is linked in [Itô/SDE author verification](../ITO-CALCULUS-SDE-VERIFICATION.md). The earlier design-phase status above is historical; the exact destination freeze is in its author packet. No unrelated topic or source mapping was changed.\n'
note.write_text(text,encoding='utf-8')
design=ROOT/'docs/teaching/ITO-CALCULUS-SDE-LESSON-DESIGN.md'
text=design.read_text(encoding='utf-8')
text=text.replace('This is the requested design-only phase, pending root design review/registration. The published body is unchanged; design readiness is not implementation, author verification or completion.','Root approved and registered this design before implementation. The complete owned source is now author-verified; exact final status, evidence and source fingerprints are in [the verification record](ITO-CALCULUS-SDE-VERIFICATION.md). The proposal wording below is retained as design history; independent review and shared production integration remain separate.')
text=text.replace('The actual destination note stays open until implementation and evidence exist.', 'The canonical destination note is now resolved against the actual implementation and [author evidence](ITO-CALCULUS-SDE-VERIFICATION.md); independent review remains separately tracked.')
text=text.replace('After design approval, the next action is complete owned lesson/model/lab/native authoring, followed by actual evidence and independent review.', 'Root approved the design before implementation; complete owned source and actual author evidence are now available for independent review.')
text += '\n\n## Implemented design decisions\n\nThe final lesson uses thirteen full programs, eleven independent tasks, five investigations and four inline figures. A centered cubic supplies the promised time-dependent transformation. The growth path uses log coordinates with ordinary-state readouts rather than a redundant second path plot; analytic mean/median positions are explicit. OU uses a probability-flow investigation; its exact joint-noise construction is supplied by the complete native example and independently checked companion model. The coupled-solver lab fixes drift/time/start to isolate resolution and noise, while the separate error investigation varies drift. Compact analytic curves and the one-step stress fixture fit narrow screens; dense paths preserve native-readable axes with local scrolling. These are reasoned representation choices, not coverage omissions or a fixed widget quota. All original coverage, exact code/output and the original extended practice survive. See the author verification for actual checks and review boundaries.\n'
design.write_text(text,encoding='utf-8')
print(json.dumps({'authorFrozenAt':now,'production':production,'programs':native['programs'],'modelComparisons':model['comparisons'],'nativeComparisons':native['numericComparisons'],'states':[r['states'] for r in browser['results']],'openedImages':len(images)},indent=2))
