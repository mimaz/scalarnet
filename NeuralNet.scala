/*
 * MIT License
 *
 * Copyright (c) 2017 Mieszko Mazurek <mimaz@gmx.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package mimaz.scalarnet

object NeuralNet {
  def sigmoid(x: Float): (Float, Float) =
    if (x > -ExpLimit) {
      val y = (1 / (1 + math.exp(-x))).toFloat
      val d = y * (1 - y)

      (y, d)
    } else {
      (0, 0)
    }

  def step(x: Float): (Float, Float) =
    (if (x < 0) 0 else 1, 0)

  def relu(x: Float): (Float, Float) =
    if (x < 0)
      (0, 0)
    else
      (x, 1)

  def leakyRelu(factor: Float)(x: Float): (Float, Float) =
    if (x < 0)
      (x * factor, factor)
    else
      (x, 1)

  def softmax(x: Float): (Float, Float) =
    if (x > ExpLimit) {
      (x, 1)
    } else if (x < -ExpLimit) {
      (0, 0)
    } else {
      val y = math.log(1 + math.exp(x)).toFloat
      val d = (1 / (1 + math.exp(-x))).toFloat

      (y, d)
    }

  def linear(x: Float): (Float, Float) =
    (x, 1)

  val DefaultLearningRate: Float = 0.125f

  val DefaultLearningMomentum: Float = 0.875f

  val DefaultActivation: Float => (Float, Float) = sigmoid

  private val ExpLimit = 40
}

class NeuralNet(topology: Seq[Int]) {
  def writeInput(data: Seq[Float]): Unit =
    _layers.head.writeValues(data)

  def readOutput(): Seq[Float] =
    _layers.last.readValues()

  def run(input: Seq[Float],
          target: Seq[Float]): Seq[Float] = {
    writeInput(input)

    forward()

    backprop(target)

    update()

    readOutput()
  }

  def run(input: Seq[Float]): Seq[Float] = {
    writeInput(input)

    forward()

    readOutput()
  }

  def reset(): Unit =
    _layers.foreach(l => l.reset())

  var learningRate: Float = NeuralNet.DefaultLearningRate

  var learningMomentum: Float = NeuralNet.DefaultLearningMomentum

  var outputActivation: Float => (Float, Float) =
    NeuralNet.DefaultActivation

  var hiddenActivation: Float => (Float, Float) =
    NeuralNet.DefaultActivation

  private lazy val _layers: IndexedSeq[Layer] = {
    for (i <- topology.indices) yield {
      def get(i: Int): Int =
        try {
          topology(i)
        } catch {
          case _: IndexOutOfBoundsException => 0
        }

      val tsiz = get(i)
      val psiz = get(i - 1)

      new Layer(tsiz, psiz)
    }
  }

  private def forward(): Unit = {
    for (i <- _layers.indices.tail.init)
      _layers(i).forward(_layers(i - 1), hiddenActivation)

    _layers.last.forward(_layers.init.last, outputActivation)
  }

  private def backprop(target: Seq[Float]): Unit = {
    _layers.last.checkout(target)

    for (i <- _layers.indices.tail.reverse)
      _layers(i).backprop(_layers(i - 1))
  }

  private def update(): Unit = {
    for (i <- _layers.indices.tail.reverse)
      _layers(i).update(_layers(i - 1))
  }

  private def allocate(amount: Int): Int = {
    val begin = _index

    _index += amount

    if (_memory.length < _index) {
      val arr = new Array[Float](_index)

      _memory.copyToArray(arr)
      _memory = arr
    }

    begin
  }

  private var _memory: Array[Float] = new Array[Float](1)

  private var _index: Int = 0

  private class Layer(val size: Int,
                      val prevSize: Int) {
    def forward(prev: Layer,
                activate: Float => (Float, Float)): Unit = {
      for (ti <- 0 until size) {
        var sum: Float = 0

        for (pi <- 0 until prevSize) {
          val inval = _memory(prev._ValuesOff + pi)
          val inwei = _memory(_WeightsOff + ti * _InputCount + pi)

          sum += inval * inwei
        }

        sum += _memory(_WeightsOff + ti * _InputCount + _InputCount)

        val Limit = 1000

        if (sum > Limit || sum.isNaN)
          sum = Limit
        else if (sum < -Limit)
          sum = -Limit

        val (y, d) = activate(sum)

        _memory(_ValuesOff + ti) = y
        _memory(_DerivativesOff + ti) = d
      }
    }

    def checkout(target: Seq[Float]): Unit =
      for ((ti, v) <- 0 until size zip target) {
        val df = v - _memory(_ValuesOff + ti)

        _memory(_GradientOff + ti) = df
      }

    def backprop(prev: Layer): Unit = {
      for (pi <- 0 until prev.size)
        _memory(prev._GradientOff + pi) = 0

      for (ti <- 0 until size) {
        val g = _memory(_GradientOff + ti)

        for (pi <- 0 until prev.size) {
          val w = _memory(_WeightsOff + ti * _InputCount + pi)

          _memory(prev._GradientOff + pi) += w * g
        }
      }

      for (pi <- 0 until prev.size)
        _memory(prev._GradientOff + pi) *= _memory(prev._DerivativesOff + pi)
    }

    def update(prev: Layer): Unit = {
      val rt = learningRate
      val mm = learningMomentum

      for (ti <- 0 until size) {
        val gr = _memory(_GradientOff + ti)

        for (pi <- 0 until prevSize) {
          val pv = _memory(prev._ValuesOff + pi)

          val did = _DeltasOff + ti * _InputCount + pi
          val wid = _WeightsOff + ti * _InputCount + pi

          _memory(did) *= mm
          _memory(did) += rt * gr * pv
          _memory(wid) += _memory(did)
        }

        val did = _DeltasOff + ti * _InputCount + prevSize
        val wid = _WeightsOff + ti * _InputCount + prevSize

        _memory(did) *= mm
        _memory(did) += rt * gr
        _memory(wid) += _memory(did)
      }
    }

    def writeValues(data: Seq[Float]): Unit =
      for ((i, v) <- (0 until size) zip data)
        _memory(_ValuesOff + i) = v

    def readValues(): Seq[Float] = {
      var vec: Vector[Float] = Vector.empty

      for (i <- 0 until size)
        vec = vec :+ _memory(_ValuesOff + i)

      vec
    }

    def reset(): Unit = {
      def random(): Float =
        (math.random() - 0.5).toFloat

      for (i <- 0 until size * _InputCount)
        _memory(_WeightsOff + i) = random()

      for (i <- 0 until size * _InputCount)
        _memory(_DeltasOff + i) = 0
    }

    private val _InputCount: Int = prevSize + 1

    private val _ValuesOff: Int = allocate(size)

    private val _DerivativesOff: Int = allocate(size)

    private val _GradientOff: Int = allocate(size)

    private val _WeightsOff: Int = allocate(size * _InputCount)

    private val _DeltasOff: Int = allocate(size * _InputCount)
  }

  reset()
}
