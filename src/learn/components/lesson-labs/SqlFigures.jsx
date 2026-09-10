import { sqlSensors, sqlReadings } from "../../data/sql-models.js";

import './scientific-concept-visuals.css';

export function SqlRelationshipDiagram() {
  return <figure className="sci-visual sci-relations" aria-label="Readings reference sensors; one sensor can have many observations">
    <figcaption><strong>One sensor can own several observations</strong><span>Follow sensor_id. A reading has its own reading_id.</span></figcaption>
    <div className="sci-relations-head"><span>sensors · unique sensor_id</span><span>readings · unique reading_id</span></div>
    {sqlSensors.map(sensor => <div className="sci-relation-lane" key={sensor.id}><div className="sci-sensor-node"><strong>{sensor.id}</strong><span>{sensor.room}</span></div><span className="sci-relation-arrow" aria-hidden="true">←</span><div className="sci-observation-branch">{sqlReadings.filter(reading => reading.sensor === sensor.id).map(reading => <div key={reading.id}><strong>r{reading.id}</strong><code>sensor_id: {reading.sensor}</code><span>{reading.minute} min · {reading.value === null ? 'NULL value' : `${reading.value} °C`}</span></div>)}{!sqlReadings.some(reading => reading.sensor === sensor.id) && <p className="sci-no-observation">No observation row exists for C.</p>}</div></div>)}
    <p className="sci-caption">Each observation's foreign key points to one sensor. Repeated A references are valid; repeated reading IDs are not. B's r3 exists with an unknown value, while C has no observation to point back from. The diagram describes the rows, not the order a database executes a join.</p>
  </figure>;
}
