"""Hand-written lecture and meeting scripts with ground-truth labels.

Why: every downstream metric needs a label the recogniser could not have seen. Each lecture
carries its sentences, the key terms a keyword extractor should surface, the sentences a summary
must keep and search queries with the sentence that answers them. Each meeting carries speaker
turns, which turns contain an action item (with owner and deadline) and search queries. The texts
use American spelling and spell out most numbers so the WER normaliser stays simple and symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Query:
    """A search query and the index of the sentence or turn that answers it."""

    text: str
    answer_index: int


@dataclass(frozen=True, slots=True)
class Lecture:
    """One single-speaker script."""

    id: str
    title: str
    sentences: tuple[str, ...]
    key_terms: tuple[str, ...]
    key_sentences: tuple[int, ...]
    queries: tuple[Query, ...]

    @property
    def text(self) -> str:
        """Full reference transcript."""
        return " ".join(self.sentences)


@dataclass(frozen=True, slots=True)
class Turn:
    """One speaker turn in a meeting."""

    speaker: str
    text: str
    action: bool = False
    owner: str | None = None
    due: str | None = None


@dataclass(frozen=True, slots=True)
class Meeting:
    """One multi-speaker script."""

    id: str
    title: str
    turns: tuple[Turn, ...]
    queries: tuple[Query, ...] = field(default_factory=tuple)

    @property
    def speakers(self) -> tuple[str, ...]:
        """Distinct speakers in order of first appearance."""
        seen: list[str] = []
        for turn in self.turns:
            if turn.speaker not in seen:
                seen.append(turn.speaker)
        return tuple(seen)

    @property
    def text(self) -> str:
        """Full reference transcript."""
        return " ".join(turn.text for turn in self.turns)


LECTURES: tuple[Lecture, ...] = (
    Lecture(
        id="photosynthesis",
        title="Photosynthesis",
        sentences=(
            "Today we are going to look at photosynthesis, the process that plants use to turn light into "
            "chemical energy.",
            "The reaction takes place inside chloroplasts, where a green pigment called chlorophyll absorbs "
            "mostly red and blue light.",
            "Carbon dioxide enters the leaf through small pores called stomata, and water travels up from the "
            "roots through the xylem.",
            "In the light dependent reactions, energy from photons splits water molecules and releases oxygen "
            "as a by-product.",
            "That energy is stored temporarily in two carrier molecules, ATP and NADPH.",
            "The Calvin cycle then uses those carriers to fix carbon dioxide into a three carbon sugar.",
            "Three turns of the cycle produce one molecule of glyceraldehyde three phosphate, which the plant "
            "later assembles into glucose.",
            "The overall equation is six carbon dioxide plus six water gives one glucose plus six oxygen.",
            "Temperature, light intensity and carbon dioxide concentration all limit the rate, and whichever "
            "is scarcest sets the ceiling.",
            "For the exam, remember that the oxygen comes from water, not from carbon dioxide, which was "
            "shown using isotope labeling.",
        ),
        key_terms=(
            "photosynthesis",
            "chlorophyll",
            "chloroplasts",
            "stomata",
            "calvin cycle",
            "carbon dioxide",
            "glucose",
            "light dependent reactions",
            "oxygen",
            "xylem",
        ),
        key_sentences=(0, 5, 7),
        queries=(
            Query("where does the oxygen come from", 9),
            Query("what limits the rate", 8),
            Query("calvin cycle carriers", 5),
        ),
    ),
    Lecture(
        id="newton",
        title="Newton's laws of motion",
        sentences=(
            "This lecture covers Newton's three laws of motion and how they describe forces acting on "
            "everyday objects.",
            "The first law says that an object keeps its velocity unless a net external force acts on it, "
            "which we call inertia.",
            "A book resting on a table stays still because the normal force from the table balances its weight.",
            "The second law relates force, mass and acceleration: the net force equals mass times acceleration.",
            "Double the force on the same mass and the acceleration doubles; double the mass and it halves.",
            "The unit of force, the newton, is the force that gives one kilogram an acceleration of one meter "
            "per second squared.",
            "The third law says that forces come in pairs: when a rocket pushes exhaust gas downward, the gas "
            "pushes the rocket upward.",
            "A common mistake is to think the action and reaction forces cancel; they act on different bodies, "
            "so they never cancel.",
            "Friction is just another force in the sum, and it always points against the relative sliding motion.",
            "In the problem set you will draw free body diagrams, label every force, and apply the second law "
            "along each axis.",
        ),
        key_terms=(
            "newton",
            "laws of motion",
            "inertia",
            "net force",
            "acceleration",
            "mass",
            "friction",
            "free body diagrams",
            "third law",
            "normal force",
        ),
        key_sentences=(1, 3, 6),
        queries=(
            Query("free body diagrams", 9),
            Query("unit of force", 5),
            Query("why action and reaction do not cancel", 7),
        ),
    ),
    Lecture(
        id="binary_search",
        title="Binary search",
        sentences=(
            "Binary search finds an item in a sorted array by repeatedly halving the part of the array that "
            "could contain it.",
            "You compare the target with the middle element, then discard the half that cannot hold the target.",
            "Each comparison removes half of the remaining candidates, so the running time is logarithmic in "
            "the array length.",
            "For a million elements that is about twenty comparisons, compared with up to a million for a linear scan.",
            "The classic bug is computing the midpoint as low plus high divided by two, which can overflow in "
            "languages with fixed width integers.",
            "The safe form is low plus the difference between high and low divided by two.",
            "Another common mistake is an off by one error in the loop condition, which makes the search miss "
            "the last element.",
            "The precondition matters: if the array is not sorted, binary search returns garbage without any warning.",
            "The same idea powers bisection on a monotonic function and the lower bound operation in standard "
            "libraries.",
            "In the lab you will implement the search, write tests for empty arrays and duplicates, and measure "
            "the comparisons.",
        ),
        key_terms=(
            "binary search",
            "sorted array",
            "midpoint",
            "logarithmic",
            "off by one",
            "overflow",
            "linear scan",
            "precondition",
            "bisection",
            "lower bound",
        ),
        key_sentences=(0, 2, 7),
        queries=(
            Query("midpoint overflow", 4),
            Query("how many comparisons for a million elements", 3),
            Query("unsorted array", 7),
        ),
    ),
    Lecture(
        id="supply_demand",
        title="Supply and demand",
        sentences=(
            "Supply and demand is the model economists use to explain how prices form in a competitive market.",
            "The demand curve slopes downward because at a higher price fewer buyers are willing and able to purchase.",
            "The supply curve slopes upward because higher prices make it worthwhile for more producers to sell.",
            "The equilibrium price is where the two curves cross, so the quantity buyers want equals the "
            "quantity sellers offer.",
            "If the price sits above equilibrium there is a surplus, and sellers cut prices to clear their stock.",
            "If the price sits below equilibrium there is a shortage, and buyers bid the price up.",
            "A shift in demand, for example from a rise in income, moves the whole curve rather than a point along it.",
            "Price elasticity measures how strongly quantity responds to a price change, and it decides who "
            "bears a tax.",
            "A price ceiling such as rent control set below equilibrium creates a persistent shortage.",
            "Next week we apply the model to the labor market, where the price is the wage.",
        ),
        key_terms=(
            "supply and demand",
            "equilibrium price",
            "demand curve",
            "supply curve",
            "surplus",
            "shortage",
            "price elasticity",
            "price ceiling",
            "rent control",
            "competitive market",
        ),
        key_sentences=(0, 3, 7),
        queries=(Query("rent control", 8), Query("what happens above equilibrium", 4), Query("price elasticity", 7)),
    ),
    Lecture(
        id="water_cycle",
        title="The water cycle",
        sentences=(
            "The water cycle describes how water moves between the oceans, the atmosphere and the land.",
            "Evaporation from the sea surface and transpiration from plants lift water vapor into the air.",
            "As the vapor rises it cools, and condensation onto dust particles forms cloud droplets.",
            "When droplets grow heavy enough they fall as precipitation, as rain, snow or hail depending on "
            "the temperature.",
            "On land, water either runs off into rivers, soaks into the ground as infiltration, or is stored as ice.",
            "Groundwater moves slowly through aquifers and can take centuries to reach the sea.",
            "The sun drives the whole cycle, and roughly ninety percent of atmospheric moisture comes from the oceans.",
            "A warmer atmosphere holds more vapor, which is why climate change intensifies both droughts and floods.",
            "Human activity changes the cycle through dams, irrigation and the paving of cities, which speeds "
            "up runoff.",
            "For the field trip, bring the rain gauge data and we will estimate the catchment water balance.",
        ),
        key_terms=(
            "water cycle",
            "evaporation",
            "transpiration",
            "condensation",
            "precipitation",
            "infiltration",
            "groundwater",
            "aquifers",
            "runoff",
            "water balance",
        ),
        key_sentences=(0, 3, 6),
        queries=(Query("aquifers", 5), Query("what drives the water cycle", 6), Query("effect of paving cities", 8)),
    ),
    Lecture(
        id="french_revolution",
        title="The French Revolution",
        sentences=(
            "The French Revolution began in 1789 and reshaped politics across Europe within a decade.",
            "France was bankrupt after funding wars, and the tax system exempted the nobility and the clergy.",
            "The king called the Estates General for the first time since 1614 to approve new taxes.",
            "The Third Estate, representing the commoners, broke away and declared itself the National Assembly.",
            "On the fourteenth of July a Paris crowd stormed the Bastille, a prison that symbolized royal power.",
            "The Declaration of the Rights of Man stated that sovereignty belongs to the nation, not the king.",
            "Radicalization followed: the monarchy was abolished in 1792 and Louis the sixteenth was executed "
            "the next year.",
            "The Reign of Terror under Robespierre executed thousands of suspected enemies of the revolution.",
            "Napoleon's coup in 1799 ended the revolutionary decade but kept many of its legal reforms.",
            "For the seminar, read the two chapters on the causes and come ready to argue whether finance or "
            "ideas mattered more.",
        ),
        key_terms=(
            "french revolution",
            "estates general",
            "national assembly",
            "bastille",
            "declaration of the rights of man",
            "reign of terror",
            "robespierre",
            "napoleon",
            "third estate",
            "monarchy",
        ),
        key_sentences=(0, 3, 5),
        queries=(
            Query("storming of the bastille", 4),
            Query("reign of terror", 7),
            Query("why was france bankrupt", 1),
        ),
    ),
)

MEETINGS: tuple[Meeting, ...] = (
    Meeting(
        id="course_staff",
        title="Course staff meeting",
        turns=(
            Turn("Priya", "Thanks for joining, let's start with the grading backlog for the second assignment."),
            Turn("Tom", "We have about forty submissions left, and the rubric for question three is still ambiguous."),
            Turn(
                "Priya",
                "Tom, please rewrite the rubric for question three and share it with the graders by Thursday.",
                action=True,
                owner="Tom",
                due="Thursday",
            ),
            Turn(
                "Mei",
                "The students also asked for a worked example of the proof, so I will record a short video this week.",
                action=True,
                owner="Mei",
                due="this week",
            ),
            Turn("Tom", "Agreed, that should cut the number of office hour questions."),
            Turn("Priya", "Next item is the midterm exam, which is scheduled for the twelfth of November."),
            Turn(
                "Mei",
                "The room holds one hundred and twenty, but we have one hundred and thirty five students enrolled.",
            ),
            Turn(
                "Priya",
                "I will ask the registrar for a second room and confirm by Monday.",
                action=True,
                owner="Priya",
                due="Monday",
            ),
            Turn("Tom", "We should also update the course page so students know where to go.", action=True),
            Turn("Mei", "Last thing, the plagiarism checker flagged two submissions that look identical."),
            Turn(
                "Priya",
                "Mei, can you send both cases to the academic integrity office before Friday?",
                action=True,
                owner="Mei",
                due="Friday",
            ),
            Turn("Mei", "Yes, I will send them tomorrow morning.", action=True, owner="Mei", due="tomorrow"),
            Turn("Priya", "Great, that is everything, thanks everyone."),
        ),
        queries=(
            Query("midterm room capacity", 6),
            Query("plagiarism checker", 9),
            Query("rubric for question three", 2),
        ),
    ),
    Meeting(
        id="research_weekly",
        title="Research group weekly",
        turns=(
            Turn("Daniel", "Let's go round the table, starting with the speech recognition experiments."),
            Turn(
                "Karen",
                "The new model cut the word error rate from twelve percent to nine percent on the noisy test set.",
            ),
            Turn("Rishi", "That is a solid gain, but we need to check whether it holds on the accented speakers."),
            Turn(
                "Karen",
                "I will rerun the evaluation on the accented subset and post the numbers in the channel by Wednesday.",
                action=True,
                owner="Karen",
                due="Wednesday",
            ),
            Turn(
                "Daniel", "The paper deadline is the third of December, so we need a full draft two weeks before that."
            ),
            Turn(
                "Rishi",
                "I will write the related work section and send it to Daniel for review next week.",
                action=True,
                owner="Rishi",
                due="next week",
            ),
            Turn(
                "Daniel", "Please also add the latency table, reviewers always ask about inference cost.", action=True
            ),
            Turn("Karen", "The cluster was down twice last week, which cost us about a day of compute."),
            Turn(
                "Daniel",
                "I will raise a ticket with the infrastructure team today.",
                action=True,
                owner="Daniel",
                due="today",
            ),
            Turn("Rishi", "One more thing, the dataset license requires attribution in the paper and the repository."),
            Turn("Karen", "Noted, we should add the citation to the readme before we release the code.", action=True),
            Turn("Daniel", "Good, let's wrap up and meet again next Tuesday."),
        ),
        queries=(Query("word error rate improvement", 1), Query("paper deadline", 4), Query("cluster was down", 7)),
    ),
    Meeting(
        id="product_sync",
        title="Product planning sync",
        turns=(
            Turn("Aisha", "Welcome everyone, today we need to decide the scope of the October release."),
            Turn("Ben", "The dashboard redesign is almost done, but the export feature still fails on large files."),
            Turn(
                "Carlos", "The failure comes from loading the whole file into memory, so we need to stream it instead."
            ),
            Turn(
                "Aisha",
                "Carlos, please fix the export to stream the data and have a patch ready for review by Thursday.",
                action=True,
                owner="Carlos",
                due="Thursday",
            ),
            Turn("Ben", "I will update the release notes once the fix lands.", action=True, owner="Ben"),
            Turn("Aisha", "Support reported that customers cannot find the billing page after the navigation change."),
            Turn("Carlos", "We could add a link in the account menu, that is a one line change."),
            Turn(
                "Aisha",
                "Let's do that, Ben, can you make sure the link is in the next build?",
                action=True,
                owner="Ben",
            ),
            Turn("Ben", "Sure, and I will ask the design team to review the menu order.", action=True, owner="Ben"),
            Turn("Carlos", "The security audit flagged that the API keys are logged in plain text."),
            Turn(
                "Aisha",
                "That is urgent, Carlos must remove the keys from the logs before the release ships.",
                action=True,
                owner="Carlos",
                due="before the release",
            ),
            Turn("Ben", "Thanks all, that covers the agenda."),
        ),
        queries=(
            Query("export fails on large files", 1),
            Query("billing page", 5),
            Query("api keys logged in plain text", 9),
        ),
    ),
    Meeting(
        id="incident_review",
        title="Incident review",
        turns=(
            Turn("Grace", "This is the review of Tuesday's outage, which lasted forty three minutes."),
            Turn(
                "Omar",
                "The root cause was a configuration change that pointed the service at a database "
                "replica that was still syncing.",
            ),
            Turn(
                "Lena",
                "Alerts fired within two minutes, but the on-call engineer did not have access to the "
                "deployment dashboard.",
            ),
            Turn(
                "Grace",
                "Omar, please add a validation step that rejects configuration changes pointing at a "
                "replica that is behind.",
                action=True,
                owner="Omar",
            ),
            Turn(
                "Lena",
                "I will grant dashboard access to everyone on the on-call rotation by the end of the week.",
                action=True,
                owner="Lena",
                due="end of the week",
            ),
            Turn(
                "Omar",
                "We should also write a runbook for database failover, since the recovery took longer "
                "than it should have.",
                action=True,
            ),
            Turn(
                "Grace",
                "Agreed, Lena will draft the runbook and share it for comments before the next review.",
                action=True,
                owner="Lena",
                due="before the next review",
            ),
            Turn("Lena", "The customer facing status page was updated late, about twenty minutes after the alert."),
            Turn(
                "Grace",
                "I will talk to the communications team about automating the status page update.",
                action=True,
                owner="Grace",
            ),
            Turn("Omar", "No data was lost, and the replica caught up within an hour."),
            Turn("Grace", "Thanks both, we will close the incident once the actions are done."),
        ),
        queries=(
            Query("root cause", 1),
            Query("status page updated late", 7),
            Query("how long did the outage last", 0),
        ),
    ),
)


def lecture_by_id(lecture_id: str) -> Lecture:
    """Look up a lecture; raises ``KeyError`` with the known ids."""
    for lecture in LECTURES:
        if lecture.id == lecture_id:
            return lecture
    raise KeyError(f"unknown lecture {lecture_id!r}; known: {[lec.id for lec in LECTURES]}")


def meeting_by_id(meeting_id: str) -> Meeting:
    """Look up a meeting; raises ``KeyError`` with the known ids."""
    for meeting in MEETINGS:
        if meeting.id == meeting_id:
            return meeting
    raise KeyError(f"unknown meeting {meeting_id!r}; known: {[m.id for m in MEETINGS]}")
